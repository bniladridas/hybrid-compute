# HybridCompute: My Brainchild for Image Domination  

This is **HybridCompute**. I built it to smash image processing into the next dimension. No CUDA on my Mac? Fine. I’ll hack the system and make cloud GPUs my minions. It’s fast, it’s mine, and it’s here to flex.  

---

## What It Is  
- **Tile-Splitter**: My Mac rips images apart with OpenCV. ARM NEON kicks in because I don’t mess around.  
- **GPU Beast**: Cloud NVIDIA GPUs upscale those tiles with CUDA—bicubic style, no mercy.  
- **Stitch Lord**: Slaps it all back together into something you’d frame (or meme).  

Think of it as my middle finger to Apple’s “no CUDA” rule.  

---

## Why I Made It  
I’m stuck on macOS, but I crave GPU power. This is my solution: preprocess locally, upscale remotely, win always. If you’ve got an M1 and a dream, this is your ticket.  

---

## How It Runs  
1. **Mac Does the Dirty Work**: Chops your image into tiles.  
2. **Cloud GPU Goes Hard**: Upscales them with CUDA magic.  
3. **I Finish It**: Stitches the pieces into a fat, juicy high-res output.  

---

## Get It Going  

### My Mac Setup  
```bash  
# Grab OpenCV because I said so  
brew install opencv  

# Snag my code  
git clone https://github.com/bniladridas/HybridCompute.git  
cd HybridCompute  

# Build it—don’t screw this up  
mkdir build && cd build  
cmake .. -DCMAKE_BUILD_TYPE=Release  
make -j4  
```

### Cloud GPU Setup  
```bash  
# Jump to the GPU zone  
cd cloud_gpu  

# Compile my CUDA weapon  
nvcc upscale.cu -o upscaler -lopencv_core -lopencv_imgcodecs  
```

---

## Use It Like I Do  

### Chop It  
```bash  
./preprocess my_image.jpg tiles/  
# My Mac’s fans will scream. I love it.  
```

### Upscale It  
```bash  
# Send tiles to the cloud (my script’s got you)  
./scripts/transfer_tiles.sh  

# Hit the GPU hard  
cd cloud_gpu  
./upscaler  
```

### Finish It  
```bash  
python3 scripts/stitch.py --input upscaled/ --output my_masterpiece.jpg  
# If it’s blurry, you did it wrong.  
```

---

## How It Stacks Up  
| Machine         | 4K Time | My Thoughts         |  
|-----------------|---------|---------------------|  
| M1 Mac (CPU)    | 14.7s   | Decent warm-up      |  
| NVIDIA T4       | 2.3s    | My kind of speed    |  
| NVIDIA A100     | 0.9s    | Overkill, I respect it |  

---

## Under My Hood  
- **CUDA**: 16x16 blocks, memory on lock.  
- **Bicubic**: My upscale secret sauce—clean and mean.  
- **Errors**: I check them so you don’t cry.  

---

## It’s Not Perfect  
- Mac can’t CUDA (duh).  
- Bicubic’s picky with trash input.  
- Cloud costs—don’t blame me.  

---

## My Next Moves  
- Metal for my Mac’s GPU (maybe).  
- AI upscaling if I feel fancy.  
- Faster tile transfers because I’m impatient.  

---

## Mess With It  
Fork it, tweak it, break it—I dare you. Pull requests better be good or I’ll roast you.  

---

## Legal Stuff  
MIT License. It’s mine, but you can borrow it. Don’t sue me over your cloud bill.  
