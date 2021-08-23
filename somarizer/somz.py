from .som_summarizer import BaseSomSummarizer
from .som import LuptiSom
import numpy as np

class SomSummarizer(BaseSomSummarizer):

    def run(self):
        self.log("\n*** Making SOM ***\n")
        som = self.make_som(self.config['deep'])
