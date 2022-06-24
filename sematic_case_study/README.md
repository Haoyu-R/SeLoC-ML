# SeLoC-ML Case Study Engineering Project

Here we present the TIA portal engineering project files for the conveyor belt example described in the paper. 

The code is well commented to allow interested readers to easily compare the coding/implementation effort for the three different implementations: traditional approach, template approach, and semantic approach. It should be mentioned that we define LOC (Lines of Code) to include the number of lines developers need to program, as well as other configuration inputs they need to provide, for example, the user input in the Mendix App (semantic approach).

Syntax for documenting:
```
We use the following three comments to mark a region (code block) or a specific line that developers need to code using the traditional approach.
# REGION_TO_BE_EDITED_TRADITIONAL
# END_REGION_TO_BE_EDITED_TRADITIONAL
# LINE_TO_BE_EDITED_TRADITIONAL

We use the following three comments to mark a region (code block) or a specific line that developers need to code using the template approach.
# REGION_TO_BE_EDITED_TEMPLATE
# END_REGION_TO_BE_EDITED_TEMPLATE
# LINE_TO_BE_EDITED_TEMPLATE

We use the following three comments to mark a region (code block) or a specific line that developers need to code using the semantic approach.
# REGION_TO_BE_EDITED_SEMANTIC
# END_REGION_TO_BE_EDITED_SEMANTIC
# LINE_TO_BE_EDITED_SEMANTIC
```

We comment the code with the hash character `#`. But be awar of that using `#` for comment is for demostration purpose only. Some file formats, like JSON, do not support comments by design. To use these files in practice, please remove the comments.

Project Structure:
| Files         			| Comment        |
| --------------------------|:--------------:|
| 5810154a-xxxxxx.blob    	| included       |
| dataBlock.db  			| included       |
| datatypes.udt 			| included       |
| main.py 					| included       |
| npu.conf 					| included       |
| utils.py 					| some python helper functions |
| fbLogic.scl               | not included   |

⚠️  We have to exclude the `fbLogic.scl` file from the repository because it contains some sensitive information. Neverthless, readers should be able to use other files for reference.