# Deeplabv3plus


![Screenshot from 2021-06-21 14-38-55](https://user-images.githubusercontent.com/76798025/122712222-88fed580-d29e-11eb-9313-b602d8708968.png)


## 데이터 구성
```txt
/data
	/SIA
		/multi
			/gtFine
				/train
				/val
			/leftImg8bit
				/train
				/val
```

gtFine : label<br>
leftImg8bit : image

## multi train
```bash
python main_sia_3.py --model deeplabv3plus_mobilenet --separable_conv --dataset satellites_multi --enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 512 --batch_size 8 --output_stride 16 --data_root ./datasets/data/SIA/multi
```


![Screenshot from 2021-06-21 14-39-06](https://user-images.githubusercontent.com/76798025/122712227-8ac89900-d29e-11eb-8bc5-6f7bc7ce197c.png)


![Screenshot from 2021-06-21 14-39-11](https://user-images.githubusercontent.com/76798025/122712233-8e5c2000-d29e-11eb-8220-aeed9ddca0c9.png)


![Screenshot from 2021-06-21 14-39-16](https://user-images.githubusercontent.com/76798025/122712238-9025e380-d29e-11eb-8a23-03db805d73e4.png)
