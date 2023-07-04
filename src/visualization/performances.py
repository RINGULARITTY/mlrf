import click
from PIL import Image
from datetime import datetime
import json
import os
import sys
import pandas as pd
from seaborn import heatmap
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib')))
import display_lib as dl

@click.command()
def main():
    def display_metric(metric, factor=100, invert=False):
        results = {}
        for model in ["svm", "kmeans", "xgboost"]:
            model_results = os.path.join("./reports/results", f"{model}.json")
            if not dl.check_file(model_results, "models/test"):
                return
            with open(model_results, "r") as f:
                content = json.load(f)
                most_recent_key = max(content.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d %H-%M-%S"))
                accs = []
                for v in content[most_recent_key].values():
                    accs.append(v[metric])
                results[model] = accs

        indices = ["flatten", "hog", "lbp"]
        results = pd.DataFrame(results, index=indices)
        
        plt.figure(figsize=(10, 10), dpi=80)
        plt.title(f"{metric} all Models all Features")
        if invert:
            heatmap(factor * results, annot=True, fmt=".2f", cmap='RdYlGn_r')
        else:
            heatmap(factor * results, annot=True, fmt=".2f", cmap='RdYlGn')
        plt.savefig(f"./reports/figures/{metric}.png")
        plt.show()
    
    def display_graphs(graph):
        models = ["svm", "kmeans", "xgboost"]
        features = ["flatten", "hog", "lbp"]

        images_cm = [Image.open(f"./reports/figures/{model}_{feature}_{graph}.png") for model in models for feature in features]

        widths, heights = zip(*(i.size for i in images_cm))

        total_width = len(models) * max(widths)
        total_height = len(features) * max(heights)

        new_img = Image.new('RGB', (total_width, total_height))

        x_offset = 0
        y_offset = 0

        for i, img in enumerate(images_cm):
            new_img.paste(img, (x_offset, y_offset))
            x_offset += img.width
            if (i+1) % 3 == 0:
                x_offset = 0
                y_offset += img.height

        new_img.save(f"./reports/figures/{graph}.png")
        new_img.show()

    dl.init()
    dl.group_task("Create graph of metrics")
    
    dl.sub_task("Acc")
    display_metric("acc")
    dl.ok()
    
    dl.sub_task("Predict Duration")
    display_metric("predict_duration", factor=1, invert=True)
    dl.ok()

    dl.group_task("Create graph of graph")

    dl.sub_task("Confusion matrix")
    display_graphs("cm")
    dl.ok()
    
    dl.sub_task("ROC Curves")
    display_graphs("auc")
    dl.ok()


if __name__ == "__main__":
    main()