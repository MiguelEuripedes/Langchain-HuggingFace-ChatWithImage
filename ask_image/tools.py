
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

class ImageCaptionTool(BaseTool):
    name = "Image Captioner"
    # Description for the agent to know what the tool should do
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "The tool will return a simple description of the image."

    # Function that the agent will use to run the tool
    def _run(self, image_path):
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
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
    name = "Object Detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "The tool will return a list of all detected objects.  Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score." 
      
    def _run(self, image_path):
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
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
