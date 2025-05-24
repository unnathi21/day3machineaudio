def mock_resnet_classify(image_path):

    if "bird" in image_path.lower():
        return "bird"
    elif "cat" in image_path.lower():
        return "Cat"
    else:
        return "Unknown"

image_path = "sample_bird.jpg"
result = mock_resnet_classify(image_path)
print(f"Image: {image_path} → Classification: {result}")