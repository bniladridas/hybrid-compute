#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <assert.h>
#include <unistd.h> // for access

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int count_files_in_dir(const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (!dir) return -1;
    int count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) count++;
    }
    closedir(dir);
    return count;
}

int main() {
    // Create test image directory and image
    system("mkdir -p test_images");
    system("python3 ../create_test_image.py");

    // Run preprocess_c (in same directory)
    system("./preprocess_c test_images test_output_tiles");

    // Check if output directory exists
    struct stat st;
    assert(stat("test_output_tiles", &st) == 0 && S_ISDIR(st.st_mode));

    // Check number of tiles: 256x256 / 64x64 = 4x4 = 16
    int file_count = count_files_in_dir("test_output_tiles");
    assert(file_count == 16);

    // Load one tile and check dimensions
    int width, height, channels;
    unsigned char* tile = stbi_load("test_output_tiles/test_tile_0.jpg", &width, &height, &channels, 0);
    assert(tile != NULL);
    assert(width == 64);
    assert(height == 64);
    assert(channels == 3);
    stbi_image_free(tile);

    // Clean up
    system("rm -rf test_images test_output_tiles");

    printf("All preprocess_c tests passed!\n");
    return 0;
}
