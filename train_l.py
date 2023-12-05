import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pqmf import PQMF
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    DurationDiscriminator2,
    AVAILABLE_FLOW_TYPES,
    AVAILABLE_DURATION_DISCRIMINATOR_TYPES
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    subband_stft_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
import commons
from adan import Adan

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
global_step = 0

import lightning as L
from torch.utils.data import DataLoader
# Import other necessary libraries and modules

class VITS2(L.LightningModule):
    def __init__(self, hps):
        super(VITS2, self).__init__()
        self.hps = hps
        self.automatic_optimization = False
        
        if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
            print("Using mel posterior encoder for VITS2")
            posterior_channels = 80  # vits2
            hps.data.use_mel_posterior_encoder = True
        else:
            print("Using lin posterior encoder for VITS1")
            posterior_channels = hps.data.filter_length // 2 + 1
            hps.data.use_mel_posterior_encoder = False

        # some of these flags are not being used in the code and directly set in hps json file.
        # they are kept here for reference and prototyping.
        if "use_transformer_flows" in hps.model.keys() and hps.model.use_transformer_flows == True:
            use_transformer_flows = True
            transformer_flow_type = hps.model.transformer_flow_type
            print(f"Using transformer flows {transformer_flow_type} for VITS2")
            assert transformer_flow_type in AVAILABLE_FLOW_TYPES, f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
        else:
            print("Using normal flows for VITS1")
            use_transformer_flows = False

        if "use_spk_conditioned_encoder" in hps.model.keys() and hps.model.use_spk_conditioned_encoder == True:
            if hps.data.n_speakers == 0:
                raise ValueError(
                    "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model")
            use_spk_conditioned_encoder = True
        else:
            print("Using normal encoder for VITS1")
            use_spk_conditioned_encoder = False

        if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas == True:
            print("Using noise scaled MAS for VITS2")
            use_noise_scaled_mas = True
            mas_noise_scale_initial = 0.01
            noise_scale_delta = 2e-6
        else:
            print("Using normal MAS for VITS1")
            use_noise_scaled_mas = False
            mas_noise_scale_initial = 0.0
            noise_scale_delta = 0.0

        if "use_duration_discriminator" in hps.model.keys() and hps.model.use_duration_discriminator == True:
            # print("Using duration discriminator for VITS2")
            use_duration_discriminator = True

            # - for duration_discriminator2
            # duration_discriminator_type = getattr(hps.model, "duration_discriminator_type", "dur_disc_1")
            duration_discriminator_type = hps.model.duration_discriminator_type
            print(
                f"Using duration_discriminator {duration_discriminator_type} for VITS2")
            assert duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys(
            ), f"duration_discriminator_type must be one of {list(AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys())}"
            # DurationDiscriminator = AVAILABLE_DURATION_DISCRIMINATOR_TYPES[duration_discriminator_type]

            if duration_discriminator_type == "dur_disc_1":
                self.net_dur_disc = DurationDiscriminator(
                    hps.model.hidden_channels,
                    hps.model.hidden_channels,
                    3,
                    0.1,
                    gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
                )
            elif duration_discriminator_type == "dur_disc_2":
                self.net_dur_disc = DurationDiscriminator2(
                    hps.model.hidden_channels,
                    hps.model.hidden_channels,
                    3,
                    0.1,
                    gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
                )
        else:
            print("NOT using any duration discriminator like VITS1")
            self.net_dur_disc = None
            self.use_duration_discriminator = False
        
        # Initialize your model components here
        self.net_g = SynthesizerTrn(
            len(symbols),
            posterior_channels,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            mas_noise_scale_initial=mas_noise_scale_initial,
            noise_scale_delta=noise_scale_delta,
            **hps.model)
    
        self.net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    def training_step(self, batch, batch_idx):

        # Retrieve optimizers
        optim_g, optim_d, optim_dur_disc = self.optimizers()

        # Unpack the batch data
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch

        # Forward pass for the generator
        y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_) = self.net_g(
                x, x_lengths, spec, spec_lengths, speakers)

        # Mel-spectrogram conversion
        if self.hps.model.use_mel_posterior_encoder or self.hps.data.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec.float(), 
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax)
            
        y_mel = commons.slice_segments(
                mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length)
        
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hps.data.filter_length,
            self.hps.data.n_mel_channels,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            self.hps.data.mel_fmin,
            self.hps.data.mel_fmax)

        y = commons.slice_segments(
            y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size)  # slice

        # Forward pass for the discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())

        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

        if self.net_dur_disc is not None:
            y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(hidden_x.detach(), x_mask.detach(), logw_.detach(),
                                                    logw.detach())
            
            loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(
                y_dur_hat_r, y_dur_hat_g)
            loss_dur_disc_all = loss_dur_disc
            optim_dur_disc.zero_grad()
            grad_norm_dur_disc = commons.clip_grad_value_(
                    self.net_dur_disc.parameters(), None)
            loss_dur_disc_all.backward()

        optim_d.zero_grad()
        grad_norm_d = commons.clip_grad_value_(self.net_d.parameters(), None)
        loss_disc_all.backward()

        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        if self.net_dur_disc is not None:
            y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(
                hidden_x, x_mask, logw_, logw)
        
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                            z_mask) * self.hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)

        if self.hps.model.mb_istft_vits == True:
            pqmf = PQMF(y.device)
            y_mb = pqmf.analysis(y)
            loss_subband = subband_stft_loss(self.hps, y_mb, y_hat_mb)
        else:
            loss_subband = torch.tensor(0.0)

        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband
        if self.net_dur_disc is not None:
            loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
            loss_gen_all += loss_dur_gen


        optim_g.zero_grad()
        grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), None)
        loss_gen_all.backward()

        # if global_step % self.hps.train.log_interval == 0:
        #     # tensorboard

        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm,
                    loss_mel, loss_dur, loss_kl, loss_subband]


        scalar_dict = {"loss/g/total": loss_gen_all, 
                        "loss/d/total": loss_disc_all, 
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d, 
                        "grad_norm_g": grad_norm_g}
        if self.net_dur_disc is not None:
            scalar_dict.update(
                {
                    "loss/dur_disc/total": loss_dur_disc_all, 
                    "grad_norm_dur_disc": grad_norm_dur_disc})
        scalar_dict.update(
            {
                "loss/g/fm": loss_fm, 
                "loss/g/mel": loss_mel, 
                "loss/g/dur": loss_dur, 
                "loss/g/kl": loss_kl,
                "loss/g/subband": loss_subband})

        scalar_dict.update(
            {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update(
            {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update(
            {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

        # if self.net_dur_disc is not None:
        #     scalar_dict.update({"loss/dur_disc_r" : losses_dur_disc_r,
        #                         "loss/dur_disc_g" : losses_dur_disc_g,
        #                         "loss/dur_gen" : loss_dur_gen})

        # image_dict = {
        #     "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
        #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
        #     "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        #     "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
        # }

        self.log_dict(scalar_dict)
        # self.log_dict(image_dict)
        self.log("global step", global_step)
            

    def validation_step(self, batch, batch_idx):
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch

        x, x_lengths = x, x_lengths
        spec, spec_lengths = spec, spec_lengths
        y, y_lengths = y, y_lengths
        speakers = speakers

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]

        y_hat, y_hat_mb, attn, mask, * \
            _ = self.net_g.infer(x, x_lengths, speakers, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * self.hps.data.hop_length

        if self.hps.model.use_mel_posterior_encoder or self.hps.data.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec,
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            self.hps.data.filter_length,
            self.hps.data.n_mel_channels,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            self.hps.data.mel_fmin,
            self.hps.data.mel_fmax
        )
        image_dict = {
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        }
        audio_dict = {
            "gen/audio": y_hat[0, :, :y_hat_lengths[0]]
        }
        if global_step == 0:
            image_dict.update(
                {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
            audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})

        # self.log_dict(audio_dict)
        # self.log_dict(image_dict)

    def configure_optimizers(self):
        self.optim_g = Adan(self.net_g.parameters(), 
                     lr=self.hps.train.learning_rate, 
                     weight_decay=0.02, 
                     betas=self.hps.train.betas, 
                     eps = self.hps.train.eps, 
                     max_grad_norm=0., 
                     no_prox=False)

        self.optim_d = Adan(self.net_d.parameters(), 
                        lr=self.hps.train.learning_rate, 
                     weight_decay=0.02, 
                     betas=self.hps.train.betas, 
                     eps = self.hps.train.eps, 
                     max_grad_norm=0., 
                     no_prox=False)


        if self.net_dur_disc is not None:
            self.optim_dur_disc = Adan(self.net_dur_disc.parameters(), 
                        lr=self.hps.train.learning_rate, 
                     weight_decay=0.02, 
                     betas=self.hps.train.betas, 
                     eps = self.hps.train.eps, 
                     max_grad_norm=0., 
                     no_prox=False)
        else:
            self.optim_dur_disc = None
        
        # Schedulers

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, 
            gamma=self.hps.train.lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=self.hps.train.lr_decay)
        if self.net_dur_disc is not None:
            self.scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(self.optim_dur_disc, 
                                    gamma=self.hps.train.lr_decay)
        else:
            self.scheduler_dur_disc = None

        return [self.optim_g, self.optim_d, self.optim_dur_disc], \
            [self.scheduler_g, self.scheduler_d, self.scheduler_dur_disc]

def create_train_dataloader(hps):
    # Load training data
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)

    # Custom collate function
    collate_fn = TextAudioSpeakerCollate()

    return DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=hps.train.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
        )
    
def create_validation_dataloader(hps):
    collate_fn = TextAudioSpeakerCollate()
    eval_dataset = TextAudioSpeakerLoader(
        hps.data.validation_files, hps.data)
    return DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                batch_size=hps.train.batch_size, pin_memory=True,
                                drop_last=False, collate_fn=collate_fn)


def main():
    hps = utils.get_hparams()

    

    # Initialize your model with hyperparameters
    model = VITS2(hps)

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=hps.train.epochs,
        limit_val_batches=5,
        default_root_dir=hps.model_dir
    )
    
    train_loader = create_train_dataloader(hps)
    valid_loader = create_validation_dataloader(hps)

    # Train the model
    trainer.fit(model, 
                train_loader,
                valid_loader
                )

    # Optionally, you can also run validation after training
    # If you have a separate validation dataloader in your DataModule
    # trainer.validate(model, datamodule=data_module)

if __name__ == '__main__':
    main()
