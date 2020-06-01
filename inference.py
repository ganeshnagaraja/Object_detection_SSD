from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import json
import argparse
import oyaml
from attrdict import AttrDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run training of outlines prediction model')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = oyaml.load(fd)  # Returns an ordered dict. Used for printing

config = AttrDict(config_yaml)


# Load model checkpoint
checkpoint = config.inference.checkpoint
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    # font = ImageFont.truetype("/home/gani/Desktop/calibril.ttf", 15)
    font = ImageFont.load_default()
    # arial

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw
    return annotated_image, len(det_boxes)

def save_result_with_boxes_and_inference_details(test_img_directory, results_dir=None):
    """  Itereate over all the images in the directory and run inference, save the image and create a json file in results directory
    Args:
        test_img_directory ([String]): [Path to the directory contatining the test images]
        results_dir ([String]): [Path to the directory containing results, if empty create one]
    """
    inference_details = []
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for image in os.listdir(os.path.abspath(test_img_directory)):
        original_image = Image.open(os.path.join(test_img_directory, image), mode='r')
        original_image = original_image.convert('RGB')
        annotated_image, number_of_products = detect(original_image, min_score=0.4, max_overlap=0.1, top_k=500)  #.show()
        results_file_name = results_dir + '/' + '{}-result.jpg'.format(image)
        annotated_image.save(results_file_name, 'JPEG')
        # save inference details in a json file
        inference_details.append({image: int(number_of_products)})
    return inference_details



if __name__ == '__main__':
    img_path = config.inference.image_folder
    results_dir = config.inference.results_dir
    # Save Inference results to the specified directory
    print('saving images and json file to {}'.format(results_dir))
    inference_result = save_result_with_boxes_and_inference_details(img_path, results_dir)
    with open(os.path.join(results_dir, 'image2products.json'), 'w') as j:
        json.dump(inference_result, j)

