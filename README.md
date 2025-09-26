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

This project revisits the demand-based asset pricing framework of Koijen & Yogo (2019). While their work establishes that latent demand is a key driver of stock return volatility, it leaves the economic origins of this demand unexplained.

- **Research Question:** Can a broad set of firm characteristics from the empirical asset pricing literature explain the latent demand component identified by Koijen & Yogo (2019)?
- **Key Findings:**
  - Incorporating 60+ additional firm characteristics does not systematically reduce the importance of latent demand
  - The characteristics-based approach fails to adequately capture investor demand or account for stock return volatility
  - A primary limitation is the use of 13F portfolio data, which masks fundamental investor heterogeneity
- **Implications:** The study calls for a rethinking of demand measurement, suggesting future work focus on improved investor clustering, sentiment-based factors, and endogenous supply

### 2. The Trouble with Heterogeneity: A Guide for Models in Macroeconomics and Finance

**Folder:** `Trouble-with-Heterogeneity`

This paper maps the landscape of heterogeneous agent models, critiquing the common oversimplification of agent types and evaluating advanced alternatives.

- **Research Question:** What are the capabilities and limitations of different frameworks (simple multi-agent, Mean-Field Games/HANK, large N-player games) for modeling heterogeneity in macro-finance?
- **Key Contributions:**
  - **Critique of MFGs/HANK:** While excellent for modeling rich heterogeneity, their reliance on *exogenous* aggregate risk limits their usefulness for studying the *determinants* of risk premia
  - **Advocacy for Granular Models:** Highlights that large N-player games (à la Gabaix, 2011) provide microfoundations for *endogenous* aggregate risk, making them ideal for finance applications
  - **Computational Evaluation:** Shows that naive Physics-Informed Neural Networks (PINNs) struggle with financial control problems and advocates for more robust actor-critic methods
- **Implications:** Provides a clear guide for researchers selecting a modeling framework, emphasizing the trade-offs between realism, tractability, and the ability to study endogenous risk

## Synergy Between Projects

The connection between the two projects is direct:

- **Project 1** empirically demonstrates that ignoring granular heterogeneity leads to incomplete and potentially misleading conclusions in a top-demand model
- **Project 2** provides the theoretical and computational roadmap for building the next generation of models that can properly incorporate this heterogeneity to better understand asset prices and aggregate risk
