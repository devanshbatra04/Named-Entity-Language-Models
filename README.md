# Named-Entity-Language-Models
Re-implementation of the paper Building Language Models for Text with Named Entities (ACL 2018)

Original code of the paper is available at this [GitHub repo](github.com/uclanlp/NamedEntityLanguageModel)

This effort was made in order to clean up the code and make it suitable for easy re-use. An issue that still remains is that PyTorch versions >= 1.0 are not supported. 

## Usage
##### 1. Setting the data path accordingly, we use command ``python3 main.py`` with default params to train Entity Composite Model, and type model.

##### 2. The corresponding data are in awd-lstm-lm/data folder in the link shared above.

#### If you use this code or data or our results in your research, please cite:

```
@InProceedings{P18-1221,
  author = 	"Parvez, Md Rizwan
		and Chakraborty, Saikat
		and Ray, Baishakhi
		and Chang, Kai-Wei",
  title = 	"Building Language Models for Text with Named Entities",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"2373--2383",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1221"
}
```
