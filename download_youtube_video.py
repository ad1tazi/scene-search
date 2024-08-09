import yt_dlp
import os

def download_youtube_video(url):
    ydl_opts = {
        'outtmpl': '%(title)s.%(ext)s',
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading video from: {url}")
            ydl.download([url])
            print(f"Download complete! Video saved in: {os.getcwd()}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    download_youtube_video(video_url)