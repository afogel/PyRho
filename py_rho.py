import numpy as np
from scipy.stats import norm

def to_numeric_matrix(x):
    """Convert a list to a numeric numpy array."""
    return np.array(x, dtype=float)

def sample_weighted(data, probs, n):
    """Sample n items from data based on given probabilities."""
    return np.random.choice(data, size=n, p=probs)

def sample_contingency_table(data, n, adjust_indices=True):
    """Sample items from a contingency table."""
    ret = np.zeros(n, dtype=int)
    data = to_numeric_matrix(data)
    total = data.sum()
    probs = data / total
    samples = np.random.choice(data.size, size=n, p=probs.flatten())
    np.add.at(data.flat, samples, -1)
    ret = samples
    return ret + 1 if adjust_indices else ret

def get_boot_pvalue(distribution, result):
    """Calculate bootstrap p-value."""
    if result < np.mean(distribution):
        return 1.0
    matched = np.sum(distribution >= result)
    return matched / len(distribution)

def check_brk_combo(base_rate, precision, kappa):
    """Check combination of base rate, precision, and kappa."""
    threshold = ((2 * base_rate * kappa) - (2 * base_rate) - kappa) / (kappa - 2)
    return precision > threshold

def recall(kappa, base_rate, precision):
    """Calculate recall."""
    return kappa * precision / (2 * precision - 2 * base_rate - kappa + 2 * base_rate * kappa)

def find_valid_pk(kappa_dist, kappa_prob, precision_dist, precision_prob, base_rate):
    """Find valid (precision, kappa) pair."""
    while True:
        kappa_idx = np.random.choice(len(kappa_dist), p=kappa_prob)
        curr_kappa = kappa_dist[kappa_idx]
        prec_idx = np.random.choice(len(precision_dist), p=precision_prob)
        curr_prec = precision_dist[prec_idx]
        
        if check_brk_combo(base_rate, curr_prec, curr_kappa):
            return np.array([curr_prec, curr_kappa])
        
        precision_min = (2 * base_rate * curr_kappa - 2 * base_rate - curr_kappa) / (curr_kappa - 2)
        valid_indices = np.where(precision_dist > precision_min)[0]
        
        if valid_indices.size == 0:
            continue
        
        precision_dist = precision_dist[valid_indices]
        precision_prob = precision_prob[valid_indices]
    
def generate_kp_list(num_needed, base_rate, kappa_min, kappa_max, precision_min, precision_max, distribution_type=0, distribution_length=10000):
    """Generate list of (precision, kappa) pairs."""
    kappa_dist = np.linspace(kappa_min, kappa_max, distribution_length)
    kappa_prob = norm.pdf(kappa_dist, 0.9, 0.1) if distribution_type == 1 else np.ones_like(kappa_dist) / distribution_length

    precision_dist = np.linspace(precision_min, precision_max, distribution_length)
    precision_prob = np.ones_like(precision_dist) / distribution_length

    kp_list = np.array([find_valid_pk(kappa_dist, kappa_prob, precision_dist, precision_prob, base_rate) for _ in range(num_needed)])
    
    return kp_list

def contingency_table(precision, recall, length, base_rate):
    """Create a contingency table."""
    gold1s = max(int(round(base_rate * length)), 1)
    gold0s = length - gold1s
    tp = max(int(round(gold1s * recall)), 1)
    fp = min(gold0s, max(int(round(tp * (1 - precision) / precision)), 1))

    return np.array([[tp, gold1s - tp], [fp, gold0s - fp]])

def random_contingency_table(set_length, base_rate, kappa_min, kappa_max, min_precision=0, max_precision=1):
    """Generate a random contingency table."""
    kp = generate_kp_list(1, base_rate, kappa_min, kappa_max, min_precision, max_precision)[0]
    kappa, precision = kp[1], kp[0]
    recall_value = recall(kappa, base_rate, precision)
    return contingency_table(precision, recall_value, set_length, base_rate)

def kappa_ct(ct):
    """Calculate kappa statistic from contingency table."""
    a, b = ct[0]
    c, d = ct[1]
    total = ct.sum()
    p_observed = (a + d) / total
    p_expected = (((a + b) * (a + c) + (c + d) * (b + d)) / (total * total))
    return (p_observed - p_expected) / (1 - p_expected)

def get_hand_ct(ct, hand_set_length, hand_set_base_rate):
    """Generate contingency table for hand set."""
    positives = int(np.ceil(hand_set_length * hand_set_base_rate))
    gold1s = ct[0]
    positive_indices = sample_contingency_table(gold1s, positives, False)
    sum_ones = np.sum(positive_indices == 0)
    sum_twos = np.sum(positive_indices == 1)
    other_ct = ct.copy()
    other_ct[0, 0] -= sum_ones
    other_ct[0, 1] -= sum_twos

    other_indices = sample_contingency_table(other_ct, hand_set_length - positives, False)
    all_indices = np.concatenate((positive_indices * 2, other_indices))

    return np.array([[np.sum(all_indices == 0), np.sum(all_indices == 2)], [np.sum(all_indices == 1), np.sum(all_indices == 3)]])

def get_hand_kappa(ct, hand_set_length, hand_set_base_rate):
    """Calculate kappa for hand set contingency table."""
    new_ct = get_hand_ct(ct, hand_set_length, hand_set_base_rate)
    return kappa_ct(new_ct)

def calc_rho_c(x, base_rate, test_set_length, test_set_base_rate_inflation=0, base_set_length=10000, replicates=800, kappa_threshold=0.9, kappa_min=0.40, precision_min=0.6, precision_max=1.0, kps=None):
    """Calculate rho_c."""
    if kps is None:
        kps = generate_kp_list(replicates, base_rate, kappa_min, kappa_threshold, precision_min, precision_max)

    if kps.shape[0] < replicates:
        replicates = kps.shape[0]

    saved_kappas = np.array([get_hand_kappa(contingency_table(*kp, base_set_length, base_rate), test_set_length, test_set_base_rate_inflation) for kp in kps[:replicates]])

    return get_boot_pvalue(saved_kappas, x)
