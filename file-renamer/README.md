# File Renamer Demo

This is a simple demonstration of how to use a local LLM along with structured generation with [Outlines](https://github.com/dottxt-ai/outlines) to automatically rename directories of files. The use of structured generation allows us to specify the format of the output, and the use of a local LLM means that we don't need to worry about data privacy or API keys.

This project was inspired by a Tweet from [@_xjdr](https://twitter.com/_xjdr) requesting:

```

```

# Using the File Renamer

This is a very simple demonstration. To use it you should copy the files from `source_data` into the `demo` directory (this is because the file names will be overwritten).

Then you can run the following command with the `dir` argument set to `demo`:

```bash
python src/main.py --dir demo
```

The program will then run, read the contents of each file and then replace the filename with the LLM generated name.