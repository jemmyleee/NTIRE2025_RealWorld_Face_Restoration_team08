# [NTIRE 2025 Challenge on Real-World Face Restoration](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## How to test the baseline model?

1. `git clone https://github.com/zhengchen1999/NTIRE2025_RealWorld_Face_Restoration.git`
2. Select the model you would like to test:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
    - We provide a baseline (team00): CodeFormer (default). Switch models (default is CodeFormer) through commenting the code in [test.py](./test.py#L19). Caution: You should download the pretrained model with the link in `model_zoo/team00_CodeFormer/team00_CodeFormer.txt` (or from [CodeFormer official codebase](https://github.com/sczhou/CodeFormer)), and put the files in following structure: 
      ```
      model_zoo
      └── team00_CodeFormer
         ├── CodeFormer
         │   └── codeformer.pth
         └── facelib
            ├── detection_Resnet50_Final.pth
            └── parsing_parsenet.pth
      ```

## How to use my model?
### Environments

```sh
conda create -n Team08 python=3.10
conda activate Team08
cd NTIRE2025_RealWorld_Face_Restoration_team08
pip install -r models/team08_good/requirements.txt
```
### run
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 8
```

## How to eval images using NR-IQA metrics and facial ID?

### Environments

```sh
conda create -n NTIRE-FR python=3.8
conda activate NTIRE-FR
pip install -r requirements.txt
```

### Metrics include:
1. **NIQE**, **CLIP-IQA**, **MANIQA**, **MUSIQ**, **Q-Align**  
   - Provided by [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)

2. **FID** (using FFHQ as reference)  
   - Provided by [VQFR](https://github.com/TencentARC/VQFR)
3. **Facial ID Consistency**
   - Model Provided by [AdaFace](https://github.com/mk-minchul/AdaFace)

### Pretrained Weights
You should first create a folder named `pretrained` in the root directory and download the following weights into it:

- adaface_ir50_ms1mv2.ckpt ([Google Drive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing))
- inception_FFHQ_512.pth ([Google Drive](https://drive.google.com/drive/folders/1k3RCSliF6PsujCMIdCD1hNM63EozlDIZ?usp=sharing))
### Folder Structure
```
input_LQ_dir
├── test
│   ├── CelebA
│   ├── CelebChild-Test
│   ├── ...
├── val
│   ├── CelebA
│   ├── CelebChild-Test
│   ├── ...
    
output_dir_test
├── CelebA
├── CelebChild-Test
├──...
output_dir_val
├── CelebA
├── CelebChild-Test
├──...
```

### Command to calculate metrics

```sh
python eval.py \
--mode "test" \
--output_folder "/path/to/your/output_dir_test" \
--lq_ref_folder "/path/to/input_LQ_dir" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
--use_qalign True 
```

The `eval.py` file accepts the following 6 parameters:
- `mode`: Choose whether to test images from the `test` set or the `val` set.
- `output_folder`: Path where the restored images are saved. Subdirectories should be organized by dataset names.
- `lq_ref_folder`: Path to the LQ images provided as input to the model. This path should be the parent directory of the `test` and `val` sets.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.
- `use_qalign`: Whether to use Q-Align or not, which will consume an additional 15GB of GPU memory.

### Weighted score

We use the following equation to calculate the final weighted score: 

$$
   \text{Score} = \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right) + \frac{\text{QALIGN}}{5} + \max\left(0, \frac{100-\text{FID}}{100}\right). 
$$

The score is calculated on the averaged IQA scores on all the val/test datasets. 

## License and Acknowledgement

This code repository is release under [MIT License](LICENSE). 

Several implementations are taken from: [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), [VQFR](https://github.com/TencentARC/VQFR), [AdaFace](https://github.com/mk-minchul/AdaFace). 
