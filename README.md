# Pneumothorax-Classification-and-Segmentation

# Introduction

Pneumothorax is the medical term used for a collapsed lung. This medical condition occurs when air leaks into the space between the lung lobe and pulmonary wall. It is a rare condition, there are less than a million reported cases per year in India. Although it is a rare condition and typically resolves within weeks, emergency care is needed. The most common symptoms for a pneumothorax are sudden chest pain and shortness of breath. On some occasions, a collapsed lung can be a life-threatening event.

Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

In this project we present a Computed Aided Diagnosis(CAD) system to classify and if present, segment pneumothorax from a set of chest radiographic images. This problem can be solved by breaking the problem down into smaller subproblems, then these smaller subproblems can be solved easily by relating them to other problems that have been studied in great detail in other domains.
The task undertaken in this project can be broken down into 2 subproblems. Given a high resolution posterior-anterior chest X-ray image as an input, the algorithm needs to correctly classify the presence of pneumothorax. This sub-problem can be seen as a binary supervised learning problem. 
The second sub-problem relies upon the first sub-problem. Given a high resolution posterior-anterior chest X-ray image with at least one instance of visible pneumothorax as input, the algorithm needs to correctly segment the pneumothorax(ces). This problem can be seen as a semantic image segmentation problem.
