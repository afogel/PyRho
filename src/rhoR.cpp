#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <RcppArmadillo.h>

namespace py = pybind11;
using namespace Rcpp;
using namespace arma;

/**
 * Sample a contingency table.
 * 
 * @param xx Contingency table matrix.
 * @param n Size of the contingency table.
 * @param forR Boolean flag for R index compatibility.
 * @return arma::ivec Sampled indices from the contingency table.
 */
arma::ivec sample_contingency_table(arma::imat xx, int n, bool forR = true) {
    arma::ivec ret = arma::ivec(n);
    int maxInt, s;
    for (int w = 0; w < n; w++) {
        maxInt = accu(xx);
        s = randi<int>(distr_param(0, maxInt));
        ret[w] = (s <= xx(0) ? 0 : (s <= (xx(0) + xx(1))) ? 1 : (s <= (xx(0) + xx(1) + xx(2))) ? 2 : 3);
        xx(ret[w])--;
    }
    if (forR) return(ret + 1);
    else return(ret);
}

/**
 * Calculate the bootstrapped p-value.
 * 
 * @param distribution Vector of calculated kappas.
 * @param result Calculated kappa to compare against.
 * @return double Bootstrapped p-value.
 */
double getBootPvalue_c(arma::vec distribution, double result) {
    if (result < mean(distribution)) {
        return 1.0;
    } else {
        arma::uvec matched = find(distribution >= result);
        double rho = (matched.size() * 1.0) / distribution.size();
        return rho;
    }
}

/**
 * Validate a Baserate/Kappa combo against the supplied Precision.
 * 
 * @param BR Baserate.
 * @param P Precision.
 * @param K Kappa.
 * @return bool True if the combo is valid, else false.
 */
bool check_BRK_combo(double BR, double P, double K) {
    double right = ((2 * BR * K) - (2 * BR) - K) / (K - 2);
    return (P > right);
}

/**
 * Calculate recall from kappa, BR, and P.
 * 
 * @param kappa Kappa value.
 * @param BR Baserate.
 * @param P Precision.
 * @return double Calculated recall.
 */
double recall(double kappa, double BR, double P) {
    double top = kappa * P;
    double R = top / (2 * P - 2 * BR - kappa + 2 * BR * kappa);
    return R;
}

/**
 * Find a valid precision and kappa combo.
 * 
 * @param kappaDistribution Distribution of kappa values.
 * @param kappaProbability Probability distribution of kappa values.
 * @param precisionDistribution Distribution of precision values.
 * @param precisionProbability Probability distribution of precision values.
 * @param baserate Baserate.
 * @return py::array_t<double> Array containing the valid precision and kappa combo.
 */
py::array_t<double> find_valid_pk(
    py::array_t<double> kappaDistribution, py::array_t<double> kappaProbability,
    py::array_t<double> precisionDistribution, py::array_t<double> precisionProbability,
    double baserate
) {
    auto kappaDist = kappaDistribution.unchecked<1>();
    auto kappaProb = kappaProbability.unchecked<1>();
    auto precDist = precisionDistribution.unchecked<1>();
    auto precProb = precisionProbability.unchecked<1>();

    int kappaSize = kappaDist.shape(0);
    int precSize = precDist.shape(0);

    arma::colvec kappaVec(kappaSize);
    arma::vec kappaProbVec(kappaSize);
    arma::colvec precVec(precSize);
    arma::vec precProbVec(precSize);

    for (int i = 0; i < kappaSize; ++i) {
        kappaVec[i] = kappaDist[i];
        kappaProbVec[i] = kappaProb[i];
    }

    for (int i = 0; i < precSize; ++i) {
        precVec[i] = precDist[i];
        precProbVec[i] = precProb[i];
    }

    arma::ivec kappaRng = regspace<arma::ivec>(0, kappaVec.size() - 1);
    arma::ivec whKappa;
    if (kappaProbVec.n_elem > 0) {
        whKappa = Rcpp::RcppArmadillo::sample(kappaRng, 1, false, kappaProbVec);
    } else {
        whKappa = Rcpp::RcppArmadillo::sample(kappaRng, 1, false);
    }
    double currKappa = kappaVec[whKappa.at(0)];

    arma::ivec precRng = regspace<arma::ivec>(0, precVec.size() - 1);
    arma::ivec whPrec = Rcpp::RcppArmadillo::sample(precRng, 1, false);
    double currPrec = precVec[whPrec.at(0)];

    if (!check_BRK_combo(baserate, currPrec, currKappa)) {
        double precisionMin = (2 * baserate * currKappa - 2 * baserate - currKappa) / (currKappa - 2);
        arma::uvec indicies = find(precVec > precisionMin);

        if (indicies.size() == 0) {
            return find_valid_pk(kappaDistribution, kappaProbability, precisionDistribution, precisionProbability, baserate);
        }

        precVec = precVec.elem(indicies);

        if (precProbVec.size() > 0) {
            precProbVec = precProbVec.elem(indicies);
        }

        precRng = regspace<ivec>(0, precVec.size() - 1);
        whPrec = Rcpp::RcppArmadillo::sample(precRng, 1, false);
        currPrec = precVec[whPrec.at(0)];
    }

    py::array_t<double> result = py::array_t<double>({2});
    auto r = result.mutable_unchecked<1>();
    r[0] = currPrec;
    r[1] = currKappa;
    return result;
}

/**
 * Generate a KP list.
 * 
 * @param numNeeded Number of KPs needed.
 * @param baserate Baserate.
 * @param kappaMin Minimum kappa value.
 * @param kappaMax Maximum kappa value.
 * @param precisionMin Minimum precision value.
 * @param precisionMax Maximum precision value.
 * @param distributionType Type of distribution (0 for normal, 1 for bell).
 * @param distributionLength Length of the distribution.
 * @return py::array_t<double> Matrix of kappa and precision values.
 */
py::array_t<double> generate_kp_list(
    int numNeeded, double baserate,
    double kappaMin, double kappaMax,
    double precisionMin, double precisionMax,
    int distributionType = 0, long distributionLength = 10000
) {
    double kappaStep = ((kappaMax - kappaMin) / (distributionLength - 1));
    colvec kappaDistribution = regspace(kappaMin, kappaStep, kappaMax);

    arma::vec kappaProbability;
    if (distributionType == 1) {
        kappaProbability = normpdf(kappaDistribution, 0.9, 0.1);
    }

    double precStep = (precisionMax - precisionMin) / (10000 - 1);
    colvec precisionDistribution = regspace(precisionMin, precStep, precisionMax);

    arma::vec precisionProbability;
    Rcpp::NumericMatrix KPs(numNeeded, 2);

    for (int i = 0; i < numNeeded; i++) {
        KPs(i, _) = find_valid_pk(
            kappaDistribution,
            kappaProbability,
            precisionDistribution,
            precisionProbability,
            baserate
        );
    }

    py::array_t<double> result = py::array_t<double>({numNeeded, 2});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < numNeeded; i++) {
        r(i, 0) = KPs(i, 0);
        r(i, 1) = KPs(i, 1);
    }
    return result;
}

/**
 * Create a contingency table.
 * 
 * @param precision Precision value.
 * @param rec Recall value.
 * @param length Length of the contingency table.
 * @param baserate Baserate value.
 * @return arma::imat Contingency table.
 */
arma::imat contingency_table(double precision, double rec, int length, double baserate) {
    int gold1s = max(NumericVector::create(round(baserate * length), 1));
    int gold0s = length - gold1s;
    int TP = max(NumericVector::create(round(gold1s * rec), 1));
    int FP = min(NumericVector::create(gold0s, max(NumericVector::create(round(TP * (1 - precision) / precision), 1))));

    arma::imat ct = arma::imat(2, 2);
    ct(0, 0) = TP;
    ct(0, 1) = gold1s - TP;
    ct(1, 0) = FP;
    ct(1, 1) = gold0s - FP;

    return ct;
}

/**
 * Create a random contingency table.
 * 
 * @param setLength Length of the contingency table.
 * @param baserate Baserate value.
 * @param kappaMin Minimum kappa value.
 * @param kappaMax Maximum kappa value.
 * @param minPrecision Minimum precision value.
 * @param maxPrecision Maximum precision value.
 * @return arma::imat Randomly generated contingency table.
 */
arma::imat random_contingency_table(
    int setLength, double baserate,
    double kappaMin, double kappaMax,
    double minPrecision = 0, double maxPrecision = 1
) {
    py::array_t<double> KP = generate_kp_list(1, baserate, kappaMin, kappaMax, minPrecision, maxPrecision);
    auto k = KP.unchecked<2>();
    double kappa = k(0, 1);
    double precision = k(0, 0);
    double rec = recall(kappa, baserate, precision);

    return contingency_table(precision, rec, setLength, baserate);
}

/**
 * Calculate kappa from a contingency table.
 * 
 * @param ct Contingency table.
 * @return double Kappa value.
 */
double kappa_ct(arma::imat ct) {
    double a = ct(0, 0); // gold 1 & silver 1
    double c = ct(0, 1); // gold 1 & silver 0
    double b = ct(1, 0); // gold 0 & silver 1
    double d = ct(1, 1); // gold 0 & silver 0
    double size = accu(ct);

    double pZero = (a + d) / size;
    double pYes = ((a + b) / size) * ((a + c) / size);
    double pNo = ((c + d) / size) * ((b + d) / size);
    double pE = (pYes + pNo);
    double k = (pZero - pE) / (1 - pE);

    return k;
}

/**
 * Get a handset from a contingency table.
 * 
 * @param ct Contingency table.
 * @param handSetLength Length of the handset.
 * @param handSetBaserate Baserate value for the handset.
 * @return arma::imat Handset.
 */
arma::imat getHand_ct(arma::imat ct, int handSetLength, double handSetBaserate) {
    int positives = ceil(handSetLength * handSetBaserate);
    arma::ivec positiveIndices;
    arma::imat otherCT(ct);

    if (positives > 0) {
        arma::irowvec gold1s = ct.row(0);
        positiveIndices = sample_contingency_table(gold1s, positives, false);

        double sumOnes = sum(positiveIndices == 0);
        double sumTwos = sum(positiveIndices == 1);
        positiveIndices *= 2;
        otherCT(0, 0) = otherCT(0, 0) - sumOnes;
        otherCT(0, 1) = otherCT(0, 1) - sumTwos;
    }

    arma::ivec otherAsVector = otherCT.as_col();
    arma::ivec otherIndices = sample_contingency_table(otherCT, handSetLength - positives, false);
    arma::ivec allIndices = join_cols<ivec>(positiveIndices, otherIndices);
    arma::imat newCT = arma::imat(2, 2);

    newCT(0, 0) = sum(allIndices == 0);
    newCT(1, 0) = sum(allIndices == 1);
    newCT(0, 1) = sum(allIndices == 2);
    newCT(1, 1) = sum(allIndices == 3);

    return newCT;
}

/**
 * Get kappa from a handset.
 * 
 * @param ct Contingency table.
 * @param handSetLength Length of the handset.
 * @param handSetBaserate Baserate value for the handset.
 * @return double Kappa value.
 */
double getHand_kappa(arma::imat ct, int handSetLength, double handSetBaserate) {
    arma::imat newCT = getHand_ct(ct, handSetLength, handSetBaserate);
    return kappa_ct(newCT);
}

/**
 * Calculate rho given the parameters of the initial set and the kappa of the observed handset.
 * 
 * @param x Observed kappa value.
 * @param OcSBaserate Baserate value.
 * @param testSetLength Length of the test set.
 * @param testSetBaserateInflation Baserate inflation value.
 * @param OcSLength Length of the original set.
 * @param replicates Number of replicates.
 * @param ScSKappaThreshold Kappa threshold value.
 * @param ScSKappaMin Minimum kappa value.
 * @param ScSPrecisionMin Minimum precision value.
 * @param ScSPrecisionMax Maximum precision value.
 * @param KPs Matrix of kappa and precision values.
 * @return double Calculated rho value.
 */
double calcRho_c(
    double x,
    double OcSBaserate,
    int testSetLength,
    double testSetBaserateInflation = 0,
    int OcSLength = 10000,
    int replicates = 800,
    double ScSKappaThreshold = 0.9,
    double ScSKappaMin = 0.40,
    double ScSPrecisionMin = 0.6,
    double ScSPrecisionMax = 1.0,
    py::array_t<double> KPs = py::array_t<double>()
) {
    if (KPs.size() == 0) {
        KPs = generate_kp_list(replicates, OcSBaserate, ScSKappaMin, ScSKappaThreshold, ScSPrecisionMin, ScSPrecisionMax);
    }

    auto k = KPs.unchecked<2>();
    int nrow = KPs.shape(0);
    if (nrow < replicates) {
        replicates = nrow;
    }

    arma::vec savedKappas = arma::vec(replicates);
    for (int KProw = 0; KProw < replicates; KProw++) {
        double precision = k(KProw, 0);
        double kappa = k(KProw, 1);
        double rec = recall(kappa, OcSBaserate, precision);

        arma::imat fullCT = contingency_table(precision, rec, OcSLength, OcSBaserate);
        double kk = getHand_kappa(fullCT, testSetLength, testSetBaserateInflation);
        savedKappas[KProw] = kk;
    }

    return getBootPvalue_c(savedKappas, x);
}

PYBIND11_MODULE(rhoR, m) {
    m.def("sample_contingency_table", &sample_contingency_table, "Sample a contingency table.");
    m.def("getBootPvalue_c", &getBootPvalue_c, "Calculate the bootstrapped p-value.");
    m.def("check_BRK_combo", &check_BRK_combo, "Check the baserate/kappa combo.");
    m.def("recall", &recall, "Calculate recall from kappa, BR, and P.");
    m.def("find_valid_pk", &find_valid_pk, "Find a valid precision and kappa combo.");
    m.def("generate_kp_list", &generate_kp_list, "Generate a KP list.");
    m.def("contingency_table", &contingency_table, "Create a contingency table.");
    m.def("random_contingency_table", &random_contingency_table, "Create a random contingency table.");
    m.def("kappa_ct", &kappa_ct, "Calculate kappa from a contingency table.");
    m.def("getHand_ct", &getHand_ct, "Get a handset from a contingency table.");
    m.def("getHand_kappa", &getHand_kappa, "Get kappa from a handset.");
    m.def("calcRho_c", &calcRho_c, "Calculate rho given the parameters of the initial set and the kappa of the observed handset.");
}
