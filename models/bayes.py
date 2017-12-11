def W(m, n, p, q, k):
    if m * 2**k <= p < (m+1) * 2**k and n * 2**k <= q < (n+1) * 2**k:
        return 1 / 2 ** (2*k)
    else:
        return 0

def map_formulation(G_0, G_k, height, width, k=2, covariance=1.0):
    # calculates -ln Pr[G_k | G_0]
    return 1 / (2*covariance) * sum(
        sum(G_k[m, n] - sum(
            sum(
                W(m, n, p, q, k) * G_0[p, q]
                for q in range(width*2**k))
            for p in range(height*2**k))
            for n in range(width))
        for m in range(height))

def gradient_prior(G_0, G_k, height, width, k=2, covariance=1.0):
    # calculates -ln Pr[G_0]
    pass