# CHTs Video Generation

## Introduction

This folder provides a simple pipeline to generate CHTs videos across training epochs (https://youtu.be/b5lLpOhb3BI?is=eZdulmFIPJibaSjd). The process closely follows the original pipeline, with the only difference being that the hyperbolic embedding component is handled separately.

## Workflow

1. **Generate adjacency matrices**
   Run the CHT code to obtain layer-wise adjacency matrices for each epoch.

2. **Create plain network visualization**
   Use `create_video.py` with the adjacency matrices.

3. **Hyperbolic embedding**

   * Merge the three layer-wise matrices into one monopartite network
   * Run `hyperbolic_embedding/panel_v5.m`
   * Recompile the MEX file locally if needed (C source provided)

4. **Generate final video**
   Combine the plain visualization and embedding results, then create the video across epochs.

## Notes


* Recompilation of MEX files may be required depending on your environment

