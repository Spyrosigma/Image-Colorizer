import matplotlib.pyplot as plt
import streamlit as st

from colorizers import *

# parser = argparse.ArgumentParser()
# parser.add_argument('-i','--img_path', type=str, default='imgs/test.jpg')
# # parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
# parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
# opt = parser.parse_args()

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

# if(opt.use_gpu):
# 	colorizer_eccv16.cuda()
# 	colorizer_siggraph17.cuda()

input_image = st.file_uploader("Upload Image : ", type=["jpg", "jpeg", "png"])

if input_image is not None:
	img = load_img(input_image)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

	img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	plt.imsave(f'eccv16.png{input_image.name}', out_img_eccv16)
	plt.imsave(f'siggraph17.png{input_image.name}', out_img_siggraph17)

	plt.figure(figsize=(12,8))
	plt.subplot(2,2,1)
	plt.imshow(img)
	plt.title('Original')
	plt.axis('off')

	plt.subplot(2,2,2)
	plt.imshow(img_bw)
	plt.title('Input')
	plt.axis('off')

	plt.subplot(2,2,3)
	plt.imshow(out_img_eccv16)
	plt.title('Output (ECCV 16)')
	plt.axis('off')

	plt.subplot(2,2,4)
	plt.imshow(out_img_siggraph17)
	plt.title('Output (SIGGRAPH 17)')
	plt.axis('off')
	plt.show()
