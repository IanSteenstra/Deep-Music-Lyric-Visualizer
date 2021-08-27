from imagenet_class_picker import ImageNetClassPicker

image_net_class_picker = ImageNetClassPicker()
image_net_class_picker.load_imagenet_class_labels("imagenet1000_clsidx_to_labels.txt")
image_net_class_picker.load_song_lyrics("green_eyes.txt")
print(set(image_net_class_picker.load_list('34_class_pick.pkl')))
# print(image_net_class_picker.get_classes_from_semantic_similarity(1))