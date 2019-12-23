## Code and data for "Learning Scalar Adjective Intensity from Paraphrases"

If you use the code or data in this repo, please cite the following paper:

```
@InProceedings{D18-1202,
  author = 	"Cocos, Anne
		and Wharton, Skyler
		and Pavlick, Ellie
		and Apidianaki, Marianna
		and Callison-Burch, Chris",
  title = 	"Learning Scalar Adjective Intensity from Paraphrases",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1752--1762",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1202"
}
```

### Directory contents

- The adjective graph that encodes adjectival paraphrases from PPDB of the form (`RB JJ1 <--> JJ2`) is under `jjgraph/`
- Code for computing a pairwise paraphrase-based intensity score is in `pairpredict/`
- Code for the downstream tasks of global ordering and indirect question answering are in `globalorder/` and `iqap/` respectively.
