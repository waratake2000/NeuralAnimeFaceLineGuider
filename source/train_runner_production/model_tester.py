import torch
import matplotlib.pyplot as plt
import cv2
import config
import numpy as np

def model_test(model,model_path,dataset_path,image_list,save_image_dir):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for image_name in image_list:
        with torch.no_grad():
            image = cv2.imread(f"{dataset_path}/{image_name}")
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (config.RESIZE, config.RESIZE))
            # shape_x,shape_y,_ = image.shape
            orig_image = image.copy()
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            outputs = model(image)
            print(outputs)
            outputs = outputs.cpu().detach().numpy()
            outputs = outputs.reshape(-1, 2)
            # keypoints = keypoints.reshape(-1, 2)
            keypoints = outputs * [config.RESIZE,config.RESIZE]
            # plt.subplot(3, 4, i+1)
            plt.imshow(orig_image, cmap='gray')
            for p in range(keypoints.shape[0]):
                plt.plot(keypoints[p, 0], keypoints[p, 1], 'r.')
                plt.text(keypoints[p, 0], keypoints[p, 1], f"{p}")
            plt.axis('off')
            plt.savefig(f"{save_image_dir}/valid_{image_name}")
            plt.show()
            plt.close()
