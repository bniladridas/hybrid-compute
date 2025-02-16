# 🚀 HybridCrunch: When Your M1 Mac and a Cloud GPU Fall in Love  
**Tagline:** *"Your Mac can’t CUDA, but it can flirt with a cloud GPU."*  

![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B)  
![CUDA](https://img.shields.io/badge/CUDA-12.0-green?logo=nvidia)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange?logo=opencv)  
![Humor](https://img.shields.io/badge/Humor-Level%2099%25-yellow)  

![hero-gif](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdG1wNnRjY2VjbmV4a2F6MWQ2NnN3b3NtYjV6Z2t5cWJ6cGt5b2Q4OSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7abKhOpu0NwenH3O/giphy.gif)  
*(Spoiler: The cloud GPU is doing all the work.)*  

---

## 🌟 **Why This Exists**  
Let’s face it: **you own a Mac**, which means you’re stuck in a toxic relationship with Apple’s “*we don’t do CUDA*” policy. But you also want NVIDIA engineers to slide into your DMs.  

**HybridCrunch** is your wingman. It’s a *hilariously over-engineered* way to:  
- **Pretend** your Mac is CUDA-compatible.  
- **Flex** that you can code in C++, CUDA, *and* write READMEs with personality.  
- **Prove** that even a $2,500 Mac can play nice with NVIDIA’s finest.  

---

## 🛠️ **What This Does (Without Judging You)**  

### **Step 1: M1 Mac (The “I Can Do It Myself” Phase)**  
- **Preprocess**: Chops images into tiny tiles.  
- **Secret Sauce**: Uses ARM NEON intrinsics to make your Mac sweat like it’s running Fortnite.  

### **Step 2: Cloud GPU (The “Actually Useful” Phase)**  
- **CUDA Upscaling**: Sends tiles to a cloud GPU that’s probably mining crypto when you’re not looking.  
- **Bicubic Magic**: Makes pixels *fancier* using math even your calculus teacher forgot.  

### **Step 3: Stitching (The “Putting It All Together” Phase)**  
- **Reassembles** the upscaled tiles into a final image.  
- **Hides** the fact that 12% of the tiles are upside down.  

---

## 🚦 **How It Works (In Case You’re a Visual Learner)**  
```ascii
[Your Mac] → "I’m a ✨creative professional✨"  
       ↓  
[Cloud GPU] → "lol CUDA go brrrrr"  
       ↓  
[Final Image] → "Look ma, no pixelation!"  
```

---

## ⚡ **Why Should You Care?**  
- **Impress NVIDIA Engineers**: They love people who bend hardware to their will.  
- **Annoy Apple Engineers**: “Look, I made your M1 talk to an NVIDIA GPU! 😈”  
- **Justify Your Cloud Bill**: “It’s for *art*, honey!”  

---

## 🧑💻 **Installation: For Humans**  

### **1. M1 Mac Setup**  
*(Because you’re too invested in the Apple ecosystem to quit now)*  
```bash  
# Install OpenCV (because we all need emotional support)  
brew install opencv --with-teeny-tiny-screams  

# Clone this repo like you’re stealing the Declaration of Independence  
git clone https://github.com/bniladridas/HybridCompute.git  
cd HybridCompute  
```

### **2. Compile the Preprocessor**  
*(Where your Mac pretends to be useful)*  
```bash  
mkdir build && cd build  
cmake .. -DCMAKE_BUILD_TYPE="FingersCrossed"  
make -j4  # -j8 if you’re feeling spicy 🌶️  
```

### **3. Cloud GPU Setup**  
*(Where you throw money at AWS)*  
1. Launch a GPU instance.  
2. Cry softly at the hourly cost.  
3. Compile the CUDA code:  
```bash  
cd cloud_gpu  
nvcc upscale.cu -o upscaler -lopencv_core -lopencv_imgcodecs  
```

---

## 🎮 **Usage: For the Impatient**  

### **Local Preprocessing**  
```bash  
# Split image into tiles  
./preprocess cat_meme.jpg tiles/  

# Watch your Mac’s fan make airplane noises ✈️  
```

### **Cloud Upscaling**  
```bash  
# Send tiles to the cloud (and your wallet to the shadow realm)  
./scripts/transfer_tiles.sh  

# Wait 3-5 business days for CUDA to work its magic  
```

### **Stitch the Final Image**  
```bash  
python3 scripts/stitch.py --input upscaled/ --output masterpiece.jpg  

# Marvel at your creation. Cry if it’s blurry.  
```

---

## 📊 **Benchmarks (Because Numbers Don’t Lie)**  

| Hardware          | Time to Upscale 4K Image | Your Emotional State       |  
|-------------------|--------------------------|----------------------------|  
| **M1 Mac (CPU)**  | 14.7 seconds             | 😊 “This is fine!”          |  
| **NVIDIA T4**     | 2.3 seconds              | 😎 “I’m basically Tony Stark.” |  
| **NVIDIA A100**   | 0.9 seconds              | 🚀 “I HAVE THE POWER OF GOD”   |  

---

## 🔍 **CUDA Nerds Only**  
- **Kernel Optimization**: Coalesced memory access, 16x16 thread blocks.  
- **Error Handling**: `CUDA_CHECK` macro validates every API call.  
- **Bicubic Math**: [See the cubic spline wizardry](cloud_gpu/upscale.cu#L42-L58).  

---

## 🚧 **Known Issues**  
- **CUDA Errors**: *“Error: GPU not found”* → Did you remember to pay AWS?  
- **Blurry Output**: Did you implement bicubic interpolation or just *vibes*?  
- **Existential Dread**: Why are we upscaling cat memes anyway?  

---

## 🌈 **Roadmap (If I Get Bored Enough)**  
- [ ] **Metal Compute Support**: Let your M1 GPU feel included.  
- [ ] **AI Upscaling**: Replace math with ✨*neural networks*✨.  
- [ ] **NFT Generator**: Because why not monetize regret?  

---

## 👏 **Contributing**  
**PRs Welcome!** Especially if:  
- You fix my garbage bicubic implementation.  
- You add memes to the documentation.  
- You explain why CUDA error messages look like eldritch runes.  

---

## 📜 **License**  
**MIT License** → Do whatever, just don’t sue me if your cloud bill rivals the GDP of a small nation.  

---

*Made with ❤️, 🧉, and a concerning amount of coffee.*  

## 🧑💻 **Updated Installation Instructions**  

### **1. M1 Mac Setup**  
*(Because you’re too invested in the Apple ecosystem to quit now)*  
```bash  
# Install OpenCV (because we all need emotional support)  
brew install opencv --with-teeny-tiny-screams  

# Clone this repo like you’re stealing the Declaration of Independence  
git clone https://github.com/bniladridas/HybridCompute.git  
cd HybridCompute  
```

### **2. Compile the Preprocessor**  
*(Where your Mac pretends to be useful)*  
```bash  
mkdir build && cd build  
cmake .. -DCMAKE_BUILD_TYPE="FingersCrossed"  
make -j4  # -j8 if you’re feeling spicy 🌶️  
```

### **3. Cloud GPU Setup**  
*(Where you throw money at AWS)*  
1. Launch a GPU instance.  
2. Cry softly at the hourly cost.  
3. Compile the CUDA code using the new CMakeLists.txt file:  
```bash  
cd cloud_gpu  
mkdir build && cd build  
cmake ..  
make -j4  # Adjust based on your instance type  
```

## 🎮 **Updated Usage Instructions**  

### **Local Preprocessing**  
```bash  
# Split image into tiles  
./preprocess cat_meme.jpg tiles/  

# Watch your Mac’s fan make airplane noises ✈️  
```

### **Cloud Upscaling**  
```bash  
# Send tiles to the cloud (and your wallet to the shadow realm)  
./scripts/transfer_tiles.sh  

# Run the CUDA code using the new CMakeLists.txt file  
cd cloud_gpu/build  
./upscaler  

# Wait 3-5 business days for CUDA to work its magic  
```

### **Stitch the Final Image**  
```bash  
python3 scripts/stitch.py --input upscaled/ --output masterpiece.jpg  

# Marvel at your creation. Cry if it’s blurry.  
```
*Made with ❤️, 🧉, and a concerning amount of coffee.*  

---

## 🧑💻 **Updated Installation Instructions**  

### **1. M1 Mac Setup**  
*(Because you’re too invested in the Apple ecosystem to quit now)*  
```bash  
# Install OpenCV (because we all need emotional support)  
brew install opencv --with-teeny-tiny-screams  

# Clone this repo like you’re stealing the Declaration of Independence  
git clone https://github.com/bniladridas/HybridCompute.git  
cd HybridCompute  
```

### **2. Compile the Preprocessor**  
*(Where your Mac pretends to be useful)*  
```bash  
mkdir build && cd build  
cmake .. -DCMAKE_BUILD_TYPE="FingersCrossed"  
make -j4  # -j8 if you’re feeling spicy 🌶️  
```

### **3. Cloud GPU Setup**  
*(Where you throw money at AWS)*  
1. Launch a GPU instance.  
2. Cry softly at the hourly cost.  
3. Compile the CUDA code using CMake:  
```bash  
make -j4  # Adjust based on your instance's capabilities  
```

---

## 🎮 **Updated Usage Instructions**  

### **Local Preprocessing**  
```bash  
# Split image into tiles  
