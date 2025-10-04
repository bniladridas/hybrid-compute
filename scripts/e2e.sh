if [ "$RUNNER_OS" == "Windows" ]; then
  source /c/Miniconda/etc/profile.d/conda.sh
  conda activate base
fi

mkdir -p test_images/tiles test_images/upscaled
python create_test_image.py

if [ "$RUNNER_OS" == "Windows" ]; then
  ./build/Release/preprocess.exe test_images test_images/tiles
else
  ./build/preprocess test_images test_images/tiles
fi

echo "Preprocess done"
cp test_images/tiles/* test_images/upscaled/
echo "Cp done"

i=0; while [ $i -le 15 ]; do mv test_images/upscaled/test_tile_$i.jpg test_images/upscaled/tile_$i.jpg; i=$((i+1)); done
echo "Mv done"

if [ "$RUNNER_OS" == "Windows" ]; then
  python scripts/stitch.py
else
  python -m coverage run --source=scripts scripts/stitch.py
fi

echo "Stitch done"

# Verify output exists
test -f test_images/final_output.jpg && echo "E2E test passed"