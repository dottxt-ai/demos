# NousCon 2024

This repo contains slides and code for Cameron Pfiffer's [NousCon 2024](https://lu.ma/zlgp0ljd?tk=p5leF2) 
talk about [Outlines](https://github.com/dottxt-ai/outlines) and [.txt](https://dottxt.co/).

It is intended to be run on [Modal](https://modal.com/) for people who do not have large GPUs, but
the code in `demo.py` can easily be adapted to use local hardware.

## Usage

```bash
pip install -r requirements.txt
```

If using Modal:

```bash
modal run demo.py
```

Modal does annoying things to the terminal output, so you may wish to disable
it by running in quiet mode:

```bash
modal run -q demo.py
```
