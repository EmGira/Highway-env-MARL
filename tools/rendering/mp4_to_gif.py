import argparse
import os
import subprocess
import sys

try:
    import imageio_ffmpeg
except ImportError:
    print("imageio_ffmpeg non è installato. Assicurati di essere nel venv corretto.")
    sys.exit(1)

def convert_mp4_to_gif(input_mp4, output_gif, speed_multiplier=1.0, fps=30):

    if not os.path.exists(input_mp4):
        print(f"Errore: Il file {input_mp4} non esiste.")
        return

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    print(f"Conversione in corso: {input_mp4} -> {output_gif}")
    print(f"Velocità: {speed_multiplier}x, FPS: {fps}")

    vf_filter = f"setpts={1.0/speed_multiplier}*PTS,fps={fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    
    cmd = [
        ffmpeg_exe,
        "-y",             
        "-i", input_mp4, 
        "-vf", vf_filter, 
        "-loop", "0",     
        output_gif
    ]

    try:
  
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        print(f"Finito! Salvato come {output_gif}\n")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante la conversione con ffmpeg:\n{e.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converti MP4 in GIF con velocità regolabile (usa pochissima RAM).")
    parser.add_argument("input", help="File MP4 di input")
    parser.add_argument("-o", "--output", help="File GIF di output (opzionale, di default usa lo stesso nome con .gif)")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Moltiplicatore di velocità (es. 1.5 o 2.0 per renderla più veloce)")
    parser.add_argument("--fps", type=int, default=30, help="Framerate della GIF (default: 30)")
    
    args = parser.parse_args()
    
    out_file = args.output
    if not out_file:
        base, _ = os.path.splitext(args.input)
        out_file = f"{base}.gif"
        
    convert_mp4_to_gif(args.input, out_file, speed_multiplier=args.speed, fps=args.fps)
