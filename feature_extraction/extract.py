import json
from .features import FeatureExtractor

if __name__ == '__main__':
    with open("Extract_features/asl.json", "r") as f:
        data = json.load(f)
    video_names = data["train"] + data["val"] + data["test"]

    variant = "new"
    if variant is None:
        underscore_variant = ""
    else:
        underscore_variant = "_" + variant

    feature_extractor = FeatureExtractor(
        video_names = video_names,
        frames_path = '/ds/videos/opticalflow-BOBSL/ASL/flow/',
        features_path = f'/ds/videos/opticalflow-BOBSL/ASL/features/extracted_features{underscore_variant}/',
        model_name = "resnet101",
        variant = variant,
    )
    feature_extractor.run()

    print("Done running feature_extractor")
