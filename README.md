<div align=center>
<img src="figures/logo2.png" width="100px">
</div>

<h2 align="center"> LlamaV-o1: Rethinking Step-By-Step Visual Reasoning in LLMs</h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>


<!-- <div style="display: flex; align-items: center; justify-content: center;">
  <img src="figures/logo2.png" width="80px" style="margin-right: 10px;">
  <h2>LlamaV-o1: Rethinking Step-By-Step Visual Reasoning in LLMs</h2>
</div> -->

## 📣 Latest Updates

- **Junuary-13-2025**: *Arxiv release. Our VCR-Bench is available at:[HuggingFace](https://huggingface.co/datasets/omkarthawakar/VRC-Bench). Model Checkpoint:[HuggingFace](https://huggingface.co/omkarthawakar/LlamaV-o1). Code is available at:[GitHub](https://github.com/mbzuai-oryx/LlamaV-o1/).🤗
--- 

## 🔥 Highlights

**LlamaV-o1** is a Large Multimodal Model capable of spontaneous reasoning.

- Our LlamaV-o1 model outperforms **Gemini-1.5-flash**,**GPT-4o-mini**, **Llama-3.2-Vision-Instruct**, **Mulberry**, and **Llava-CoT** on our proposed VCR-Bench.

- Our LlamaV-o1 model outperforms **Gemini-1.5-Pro**,**GPT-4o-mini**, **Llama-3.2-Vision-Instruct**, **Mulberry**, **Llava-CoT**, etc. on six challenging multimodal benchmarks (MMStar, MMBench, MMVet, MathVista, AI2D and Hallusion).

## Contributions 🏆
- Step-by-Step Visual Reasoning Benchmark: To the best of our knowledge, the proposed
benchmark is the first effort designed to evaluate multimodal multi-step reasoning tasks
across diverse topics. The proposed benchmark, named VRC-Bench, spans around eight
different categories (Visual Reasoning, Math & Logic Reasoning, Social & Cultural Context,
Medical Imaging (Basic Medical Science), Charts & Diagram Understanding, OCR &
Document Understanding, Complex Visual Perception and Scientific Reasoning) with over
1,000 challenging samples and more than 4k reasoning steps.
- Novel Evaluation Metric: A metric that assesses the reasoning quality at the level of
individual steps, emphasizing both correctness and logical coherence.
- Combined Multi-Step Curriculum Learning and Beam Search Approach: A multimodal rea-
soning method, named LlamaV-o1, that combines the structured progression of curriculum
learning with the efficiency of Beam Search. The proposed approach ensures incremental
skill development while optimizing reasoning paths, enabling the model to be effective in
complex multi-step visual reasoning tasks in terms of both accuracy and efficiency. Specifi-
cally, the proposed LlamaV-o1 achieves an absolute gain of 3.8% in terms of average score
across six benchmarks while being 5× faster, compared to the recent Llava-CoT.
---

### Dataset Overview
<div align=center>
<img src="figures/dataset_overview.png" width="900px">
</div>
The figure presents our benchmark structure and the comparative performance of LMMs on ReasoningChain-Bench. The dataset spans diverse domains, including mathematical, logical, and scientific reasoning, visual perception, and specialized areas such as medical imaging, cultural understanding, and document OCR. It also includes tasks like chart and diagram comprehension to test real-world applications. The bar chart compares various state-of-the-art models, showcasing final answer accuracy and step-by-step reasoning performance. Our LlamaV-o1 model surpasses GPT-4o-mini, Gemini-1.5-Flash, and Llava-CoT in complex multimodal reasoning tasks, achieving superior accuracy and logical coherence.

## Dataset Examples
<div align=center>
<img src="figures/data_examples.png" width="900px">
</div>

### Results
**Table 1:** Comparison of models based on Final Answer accuracy and Reasoning Steps performance on the proposed VRC-Bench. The best results in each case (closed-source and open-source) are in bold. Our LlamaV-o1 achieves superior performance compared to its open-source counterpart (Llava-CoT) while also being competitive against the closed-source models.

| **Model**   | **GPT-4o** | **Claude-3.5** | **Gemini-2.0** | **Gemini-1.5 Pro** | **Gemini-1.5 Flash** | **GPT-4o Mini** | **Llama-3.2 Vision** | **Mulberry** | **Llava-CoT** | **LlamaV-o1 (Ours)** |
|-------------|------------|----------------|----------------|-------------------|--------------------|----------------|--------------------|-------------|--------------|-------------------|
| **Final Answer** | 59.28      | **61.35**        | 61.16          | **61.35**         | 54.99              | 56.39          | 48.40              | 51.90       | 54.09        | **56.49**         |
| **Reasoning Steps** | **76.68**   | 72.12            | 74.08          | 72.12             | 71.86             | 74.05          | 58.37              | 63.86       | 66.21        | **68.93**         |
---

### Category wise spilt
<div align=center>
<img src="figures/results_vcrbench.png" width="900px">
</div>

