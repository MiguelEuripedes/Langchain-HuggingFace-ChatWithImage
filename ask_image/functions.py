from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection

import torch
from PIL import Image

def get_image_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    model_name = 'Salesforce/blip-image-captioning-large'
    device = 'cpu'

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # Convert image (transforming) to tensor
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

def detect_objects(image_path):
    image = Image.open(image_path).convert("RGB")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)

    # Results from the image
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes = target_sizes, threshold=0.9)[0]

    # Get the labels and resources from results
    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections

if __name__ == "__main__":
    image_path = "C:/Users/Miguel/Documents/PosGrad-Mestrado/2023/LLM-Project/Langchain-Video-Summarizer/ask_image/images/motorcycle.jpg"
    detections = detect_objects(image_path)
    print(detections)  