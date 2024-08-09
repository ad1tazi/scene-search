import asyncio
import os
import base64
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI
from dotenv import load_dotenv
import cv2

load_dotenv()

OPENAI_API_KEY = ''

openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

PROMPT_START = """You will be given an image that represents a single frame from a video. Your task is to describe what is happening in this frame accurately and in very specific detail. Here is the image:

<image>"""

PROMPT_END = """</image>

Please follow these instructions to provide a comprehensive description of the image:

1. Begin by observing the overall scene. What is the general setting or environment depicted in the image?

2. Identify and describe the main subjects or focal points in the image. These could be people, animals, objects, or any other prominent elements.

3. For any people present in the image:
   - Describe their appearance (age, gender, clothing, etc.)
   - Note their positioning and posture
   - Describe any actions they appear to be performing
   - Attempt to interpret their facial expressions or emotions, if visible

4. For any objects or elements in the scene:
   - Describe their appearance, size, and condition
   - Note their position relative to other elements in the frame
   - Mention any interaction between objects or between objects and people

5. Pay attention to and describe:
   - The lighting conditions (bright, dim, natural light, artificial light, etc.)
   - Any visible weather conditions
   - The time of day, if apparent

6. Note any text or signage visible in the image and describe its content and context.

7. Describe any apparent motion or action that seems to be occurring, keeping in mind this is a still frame from a video.

8. Mention any unusual or particularly striking elements in the image.

9. If relevant, try to infer the context or purpose of the scene (e.g., a sports event, a family gathering, a work environment, etc.)

Remember to be as specific and detailed as possible in your description. Avoid making assumptions beyond what you can directly observe in the image. If something is unclear or ambiguous, it's okay to mention that uncertainty.

After considering all these aspects, provide your comprehensive description of the image. Your description should be a cohesive narrative that covers all relevant details observed in the image, typically consisting of several paragraphs."""


async def describe_frame(frame: str) -> str:
    if frame.startswith('data:image'):
        frame = frame.split(',', 1)[1]

    # Log the base64 frame data to inspect it
    print(f"Base64 Frame Data (after split): {frame[:50]}...")

    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_START
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}"  # Ensure no space after 'base64,'
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT_END
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

def convert_image_to_base64(image: Image.Image) -> str:
    rgb_image = image.convert('RGB')
    buffered = BytesIO()
    rgb_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

def get_frames(video_path: str, frame_interval: int = 30, max_width: int = 640) -> list[tuple[str, float]]:
    """
    Given the path to a video file, return a list of tuples.
    Each tuple contains a base64 encoded image of a frame and its timestamp.
    
    :param video_path: Path to the video file
    :param frame_interval: Number of frames to skip between each captured frame
    :param max_width: Maximum width of the resized frame
    :return: List of tuples (base64_frame, timestamp)
    """
    frames = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    while True:
        success = video.grab()
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            success, frame = video.retrieve()
            if not success:
                break
            
            # Calculate and print progress
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing frames: {progress:.2f}%", end="", flush=True)
            
            # Calculate timestamp
            timestamp = frame_count / fps
            
            # Resize the frame
            height, width = frame.shape[:2]
            if width > max_width:
                scale = max_width / width
                new_height = int(height * scale)
                frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode the frame as a JPEG image (faster and smaller than PNG)
            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert the buffer to a base64 string
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            
            frames.append((base64_frame, timestamp))
        
        frame_count += 1
    
    video.release()
    print("\nFrame processing complete.")
    return frames

def get_specific_frame_at_time(video_path: str, target_time: float) -> str:
    """
    Extract a specific frame from the video at the given target time.

    :param video_path: Path to the video file
    :param target_time: Time in seconds at which to extract the frame
    :return: Base64 encoded string of the frame
    """
    video = cv2.VideoCapture(video_path)
    
    # Set the video position to the target time
    video.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
    
    # Read the frame
    success, frame = video.read()
    if not success:
        video.release()
        raise ValueError(f"Could not read frame at time {target_time} seconds")
    
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Encode the frame as a JPEG image
    _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Convert the buffer to a base64 string
    base64_frame = base64.b64encode(buffer).decode('utf-8')
    
    # Log the base64 frame data to inspect it
    print(f"Base64 Frame Data (generated): {base64_frame[:50]}...")
    
    video.release()
    return base64_frame

async def main():
    video_path = "/Users/adamtazi/Projects/misc/video-search/I Fed a Beaver to my Groundhog (emotional).mp4"
    #frames = get_frames(video_path, frame_interval=50, max_width=640)

    # 1. Get the frame at 3 minutes and 17 seconds
    target_time = 3 * 60 + 17  # 3 minutes and 17 seconds in seconds
    #target_frame = next((frame for frame, timestamp in frames if timestamp >= target_time), None)
    target_frame = get_specific_frame_at_time(video_path, target_time)

    if target_frame is None:
        print("Frame not found at the specified time.")
        return

    # 2. Get the description of the frame
    description = await describe_frame(target_frame)

    # 3. Print the timestamp and the description
    print(f"Timestamp: {target_time:.2f} seconds")
    print("Description:")
    print(description)

if __name__ == "__main__":
    asyncio.run(main())