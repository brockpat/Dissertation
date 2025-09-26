# Dissertation

This repository contains the code and documentation for the two primary projects of my dissertation. The research explores the pivotal role of investor heterogeneity and demand in shaping financial markets, critiquing existing methodologies and proposing new computational and theoretical directions.

## Overview

The two projects form a cohesive investigation into one of the most important yet challenging aspects of financial economics: moving beyond representative agent models.

1. **Project 1 ("Demand-based Asset Pricing")** is an empirical project that identifies a critical limitation in a leading demand-based asset pricing framework, tracing the problem back to unmodeled investor heterogeneity.
2. **Project 2 ("Trouble with Heterogeneity")** is a theoretical and computational project that provides a deep analysis of the frameworks available for modeling this heterogeneity, evaluating their strengths, weaknesses, and potential solutions.

Together, they argue that a granular understanding of heterogeneous investors is not just an incremental improvement but a necessary foundation for accurately modeling asset prices and aggregate risk.

## 📂 Projects 

### 1. Demand-based Asset Pricing: The Limits of Characteristics and the Latent Demand Puzzle

**Folder:** `Demand-based-asset-pricing`

This project revisits the demand-based asset pricing framework of Koijen & Yogo (2019). While their work establishes that demand is a key driver of stock return volatility, it leaves the economic origins of this demand unexplained.

- **Motivation**: Koijen & Yogo show that investor demand and the holdings composition of stocks (such as predominant institutional ownership) drive stock return volatility. However, it is unclear what drives investor's holdings demand.  
- **Approach**: We extend their model by incorporating 60 additional characteristics from the empirical asset pricing literature.  
- **Key Findings**:  
  - Latent demand remains central even after accounting for observed characteristics.  
  - Investor demand is not well captured by the characteristics-based demand equation.  
  - Significant limitations arise when using **13F portfolio data** due to investor heterogeneity and the resulting estimation issues since most investors only hold very few stocks.  
- **Future Directions**:  
  - Improved clustering methods.  
  - Sentiment-based demand factors.  
  - Incorporating endogenous stock supply into demand-based frameworks. 

### 2. The Trouble with Heterogeneity: A Guide for Models in Macroeconomics and Finance

**Folder:** `Trouble-with-Heterogeneity`

This project examines how heterogeneity is vital for models in macroeconomics & finance, describes the limitations of current state-of-the-art models in this area and outlines a roadmap for future research on improving models in macroeconomics & finance.  

- **Research Question:** What are the capabilities and limitations of different frameworks (simple multi-agent, Mean-Field Games/HANK, large N-player games) for modeling heterogeneity in macroeconomics and finance?
- **Key Contributions:**
  - **Limitations of MFGs/HANK:** While excellent for modeling rich heterogeneity, their reliance on *exogenous* aggregate risk limits their usefulness for studying the *determinants* of quantities such as risk premia
  - **Advocacy for Granular Models:** Highlights that large N-player games (à la Gabaix, 2011) provide microfoundations for *endogenous* aggregate risk, making them ideal for finance applications
  - **Computational Evaluation:** Shows that naive Physics-Informed Neural Networks (PINNs) struggle with financial control problems and advocates for more robust actor-critic methods
- **Implications:** Provides a clear guide for researchers selecting a modeling framework, emphasizing the trade-offs between realism, tractability, and the ability to study endogenous risk

## Synergy Between Projects

The connection between the two projects is direct:

- **Project 1** empirically demonstrates that ignoring granular heterogeneity leads to incomplete and potentially misleading conclusions in a top-demand model
- **Project 2** provides the theoretical and computational roadmap for building the next generation of models that can properly incorporate this heterogeneity to better understand asset prices and aggregate risk
