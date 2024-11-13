from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import os

def create_narrative_video(images, audios, output_path):
    """
    Ensambla un video narrativo a partir de imágenes y audios sincronizados con transiciones.
    
    Args:
    - images: lista de rutas de archivos de imágenes en el orden narrativo.
    - audios: lista de rutas de archivos de audio correspondientes a cada imagen.
    - output_path: ruta del archivo de salida del video.
    """
    video_clips = []

    for i, (image_path, audio_path) in enumerate(zip(images, audios)):
        # Cargar el clip de imagen y el clip de audio
        img_clip = ImageClip(image_path)
        audio_clip = AudioFileClip(audio_path)
        
        # Duración de la imagen en pantalla igual a la duración del audio
        img_duration = audio_clip.duration
        
        # Calcular los tiempos de desvanecimiento de entrada y salida
        fade_duration = 0.1 * img_duration  # 10% de la duración del audio
        
        # Configurar el clip de imagen con los desvanecimientos
        img_clip = img_clip.set_duration(img_duration).fadein(fade_duration).fadeout(fade_duration)
        
        # Sincronizar el audio con la imagen
        img_clip = img_clip.set_audio(audio_clip)
        
        # Agregar el clip con la imagen y el audio sincronizados
        video_clips.append(img_clip)
        
        # Añadir un clip en negro de 1 segundo entre imágenes, excepto después de la última
        if i < len(images) - 1:
            pause_clip = ImageClip(img='black.png').set_duration(1)
            video_clips.append(pause_clip)

    # Concatenar todos los clips en un solo video
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    # Guardar el video generado
    final_video.write_videofile(output_path, codec="libx264", fps=24)
