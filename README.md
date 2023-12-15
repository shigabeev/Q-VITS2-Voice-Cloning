# The ultimate VITS2 

![Alt text](resources/image6.png)

The idea for this repo is to implement the most comprehensive VITS2 out here. 


## Changelist
- [x] Bump Librosa and python version to the highest
- [x] Implement d-vector instead of speaker id for external speaker encoder as in YourTTS.
- [x] Implement YourTTS styled d-vector-free text encoder and d-vector as an input to vocoder (currenlty only HiFiGAN does that)
- [x] implement dataloader that would load d-vectors
- [x] Add quantized Text Encoder. BERT -> bottleneck -> text features.
- [ ] VCTK audio loader
- [ ] Implement a better vocoder with support for d-vector
- [ ] Remove boilerplate code in attentions.py and replace it with native torch.nn.Encoder
- [x] Adan optimizer
- [x] PyTorch Lightning support
- [ ] Add [Bidirectional Flow Loss](https://github.com/PlayVoice/vits_chinese/issues/33)

## pre-requisites
1. Python >= 3.8
2. CUDA
3. [Pytorch](https://pytorch.org/get-started/previous-versions/#v1131) version 1.13.1 (+cu117)
4. Clone this repository
5. Install python requirements.
   ```
   pip install -r requirements.txt
   ```
   
   If you want to proceed with those cleaned texts in [filelists](filelists), you need to install espeak.
   ```
   apt-get install espeak
   ```
7. Prepare datasets & configuration
   1. wav files (22050Hz Mono, PCM-16) 
   2. Prepare text files. One for training<sup>[(ex)](filelists/ljs_audio_text_train_filelist.txt)</sup> and one for validation<sup>[(ex)](filelists/ljs_audio_text_val_filelist.txt)</sup>. Split your dataset to each files. As shown in these examples, the datasets in validation file should be fewer than the training one, while being unique from those of training text.
      
      - Single speaker<sup>[(ex)](filelists/ljs_audio_text_test_filelist.txt)</sup>
      
      ```
      wavfile_path|transcript
      ```
      

      - Multi speaker<sup>[(ex)](filelists/vctk_audio_sid_text_test_filelist.txt)</sup>
      
      ```
      wavfile_path|speaker_id|transcript
      ```
   4. Run preprocessing with a [cleaner](text/cleaners.py) of your interest. You may change the [symbols](text/symbols.py) as well.
      - Single speaker
      ```
      python preprocess.py --text_index 1 --filelists PATH_TO_train.txt --text_cleaners CLEANER_NAME
      python preprocess.py --text_index 1 --filelists PATH_TO_val.txt --text_cleaners CLEANER_NAME
      ```
      
      - Multi speaker
      ```
      python preprocess.py --text_index 2 --filelists PATH_TO_train.txt --text_cleaners CLEANER_NAME
      python preprocess.py --text_index 2 --filelists PATH_TO_val.txt --text_cleaners CLEANER_NAME
      ```
      The resulting cleaned text would be like [this(single)](filelists/ljs_audio_text_test_filelist.txt.cleaned). <sup>[ex - multi](filelists/vctk_audio_sid_text_test_filelist.txt.cleaned)</sup> 
      
9. Build Monotonic Alignment Search.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```
8. Edit [configurations](configs) based on files and cleaners you used.

## Setting json file in [configs](configs)
| Model | How to set up json file in [configs](configs) | Sample of json file configuration|
| :---: | :---: | :---: |
| iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ``` | istft_vits2_base.json |
| MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | mb_istft_vits2_base.json |
| MS-iSTFT-VITS2 | ```"subbands": 4,```<br>```"ms_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ms_istft_vits2_base.json |
| Mini-iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,``` | mini_istft_vits2_base.json |
| Mini-MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,```<br>```"upsample_initial_channel": 256,``` | mini_mb_istft_vits2_base.json |

## Training Example
```sh
# train_ms.py for multi speaker
# train_l.py to use Lightning
python train_ms.py -c configs/shergin_d_vector_hfg.json -m models/test
```
## Contact
If you have any questions regarding how to run it, contact us in Telegram

https://t.me/voice_stuff_chat

## Credits
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [ORI-Muchim/PolyLangVITS](https://github.com/ORI-Muchim/PolyLangVITS)
- [tonnetonne814/MB-iSTFT-VITS-44100-Ja](https://github.com/tonnetonne814/MB-iSTFT-VITS-44100-Ja)
- [misakiudon/MB-iSTFT-VITS-multilingual](https://github.com/misakiudon/MB-iSTFT-VITS-multilingual)
