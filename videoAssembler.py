from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

def create_narrative_video(images, audios, output_path):
    video_clips = []

    for i, (image_path, audio_path) in enumerate(zip(images, audios)):
        img_clip = ImageClip(image_path)
        audio_clip = AudioFileClip(audio_path)
        
        img_duration = audio_clip.duration
        
        fade_duration = 0.1 * img_duration
        
        img_clip = img_clip.set_duration(img_duration).fadein(fade_duration).fadeout(fade_duration)
        
        img_clip = img_clip.set_audio(audio_clip)
        
        video_clips.append(img_clip)
        
        if i < len(images) - 1:
            pause_clip = ImageClip(img='black.png').set_duration(1)
            video_clips.append(pause_clip)

    final_video = concatenate_videoclips(video_clips, method="compose")
    
    final_video.write_videofile(output_path, codec="libx264", fps=24)
