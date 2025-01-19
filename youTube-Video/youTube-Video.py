# from pytube import YouTube
#
# def download_youtube_video(url, save_path='downloads/'):
#     # Create YouTube object
#     yt = YouTube(url)
#
#     # Get the highest resolution stream
#     stream = yt.streams.get_highest_resolution()
#
#     # Download the video
#     print(f"Downloading {yt.title}...")
#     stream.download(output_path=save_path)
#     print(f"Download completed! Video saved to {save_path}")
#
# # Example usage
# youtube_url = 'https://www.youtube.com/watch?v=dBwZZXMzRuU'
# download_youtube_video(youtube_url)


import yt_dlp

def download_youtube_video(url, save_path=r'D:\Health Related\Breath\Carl Stough//'):
    ydl_opts = {
        'format': 'best',  # Download the best quality available
        'outtmpl': f'{save_path}%(title)s.%(ext)s',  # Output file template
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading video from {url}...")
        ydl.download([url])
        print(f"Download completed! Video saved to {save_path}")

# Example usage
youtube_url = 'https://www.youtube.com/watch?v=1OgPROMoAfY'
download_youtube_video(youtube_url)
