# Deep Music & Lyric Visualizer
This visualizer uses BigGAN (Brock et al., 2018), Sentiment Analysis, and Semantic Similarity to visualize music and its lyrics.
Recognition to the repo: https://github.com/msieg/deep-music-visualizer

## Summary
The code I added decides which images from the ImageNet dataset to include in the video by finding either the Sentiment or Semantic similarity between the image labels and the lyrics. 

## Testing
1. Follow the guide from the this repo: https://github.com/msieg/deep-music-visualizer.
2. Add an mp3 song to the root folder (e.g. beethoven.mp3), as well as a text file (e.g. lyrics.txt) of the lyrics of the song.
3. Run the following: ```python visualize.py --song beethoven.mp3 --class_picker 3 --lyrics_file lyrics.txt```
4. ```class_picker``` decides which type of method to use for picking the classes for the video. (1=Previous Classes; 2=Sentiment Analysis; 3=Contextual Similarity; 4=Sentiment + Contextual; 5=From Pickle File; Other=Random)
