from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import uuid

class ImageProcessor:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def text2image(self, prompt):
        image = self.pipe(prompt).images[0]
        image_filename = f"generated_image_{uuid.uuid4()}.png"
        image.save(image_filename)
        return image_filename

    def image2text(self, image_path):
        image = Image.open(image_path)
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values, max_length=16, num_beams=4, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

    def image2image(self, image_path, prompt):
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        images = self.img2img_pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
        output_path = f"img2img_output_{uuid.uuid4()}.png"
        images[0].save(output_path)
        return output_path