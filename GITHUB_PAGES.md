# Project: Time-Series-Uncertainty

This repository hosts the code and documentation for uncertainty estimation in the mean of correlated signals.

## GitHub Pages

The documentation is published using GitHub Pages from the `docs/` directory on the `main` branch.

- Main site: https://paulnorman.github.io/Time-Series-Uncertainty/
- Custom domain: https://paulnorman.cc/ (configure in repository settings)

## Publishing setup
- The static site is built and committed to the `docs/` folder.
- A `.nojekyll` file is present to disable Jekyll processing.
- To update the site, push changes to the `main` branch.

## Custom domain
To use a custom domain (e.g., `paulnorman.cc`):
1. Add a file named `CNAME` in the `docs/` directory with your domain name:
   ```
   paulnorman.cc
   ```
2. Configure your DNS to point to GitHub Pages IPs (see GitHub Pages docs).
3. Enable "Enforce HTTPS" in the repository settings after DNS propagates.

## Troubleshooting
- If HTTPS is unavailable, check DNS configuration and wait for propagation.
- For more, see: https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site
