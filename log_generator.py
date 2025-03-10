from diffusers import StableDiffusionPipeline
import torch

# Function to generate logo
def generate_logo(prompt, output_filename="generated_logo.png"):
    print("🔄 Loading Stable Diffusion model...")

    # Load model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="./models")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    print("✅ Model loaded successfully!")

    # Generate logo
    print(f"🖌 Generating logo for: {prompt}")
    image = pipe(prompt, num_inference_steps=30).images[0]

    # Save output
    image.save(output_filename)
    print(f"🎉 Logo saved as {output_filename}")

# Get user input
user_prompt = input("Enter logo description (e.g., 'A modern AI logo in blue and white'): ")
generate_logo(user_prompt)
