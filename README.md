# MLQTransfer

In this work we will make use of different machine learnings techniques, such as reinforcement learning, in order to optain optimal control pulses for the transport of quantum information across a quantum bus. The final pulse are required to give rise high fidelity protocol so the dynamic is fast enought so relaxation and dephasing times are much smaller than the transfer time. Furthermore, the pulses must be robust againts different noise sources such as a systematic shift in the applied driving pulses, or a stochastic error from external environmental noise. The pulses obtained with machine learning techniques will be validated and compared with other classical quantum control protocols such as COBYLA, CRAB or GRAPE.

## Dependences

All the numerical simulations are written in python, with the following package dependences:
```
python==3.9.16
```

## Folders structure
- Codes (main .py and jupyter notebooks)
  - Data (.npy and .npz files with saved data to plot in the report)
- Report (main .tex and .bib documents)
  - Sections (.tex for each section)
  - Figures (figures in .pdf or .png is needed)
  - Vesions (main .pdf with the major versions of the report)


## ⚠️ Warning ⚠️

The code is still in production, and several fails and bugs can be present at thed moment.

## Authors
- Pablo Ernesto Soëtard García
- David Fernández Fernández
- Gloria Platero Coello
