# Nerotiq

## Project structure

 - `src` - contains all source code.
   - `opencl` - contains code for opencl kernels, etc.
     - `activation` - activation functions and derivatives.
     - `core` - kernels for different network layer types.
     - `shared` - common shared libraries.
   - `dotnet` - contains all dotnet librares, tests, etc.
     - `Nerotiq` - main library targeting dotnet standard.
     - `Nerotiq.Test` - tests go here.

