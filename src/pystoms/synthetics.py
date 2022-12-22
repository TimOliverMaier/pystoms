from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import factorial
from scipy.stats import rv_continuous
from typing import Union


class IsotopicAveragineDistribution(rv_continuous):
    r"""Isotopic pattern distribution according to averagine-like model


    Subclass of ``scipy.rv_continuous`` with overwritten ``.pdf()``,``._pdf()``,
    ``.rvs()`` and ``._rvs()`` methods.
    Isotopic distribution is modeled as a mixture of a set of n
    normal distributions around the monoisotopic and isotopic mass-to-charge ratios.
    The weights are determined by the averagine-like model by Breen `et. al` [1].
    The underlying probability density is given by:

    .. math::

        f(x)=\frac{1}{\sigma_i\sqrt{2\pi}}\sum_{i=1}^{n}w_i e^{-0.5\left(\frac{x-\mu_i}{\sigma_i}\right)^2}

    Attributes:

        decoy (bool, optional): If False, weights are calculated according to [1].
            Else, the order of the weights is reversed. Defaults to False.

    Examples:

        This class can be instantiated without any arguments, except
        if instance shall describe decoy distributions:

        >>> iso = IsotopicAveragineDistribution()
        >>> iso_decoy = IsotopicAveragineDistribution(decoy = True)

        Parameters of distribution are given to ``.pdf()``,``.rvs()`` methods:

        >>> x = np.arange(2200,2300)/10
        >>> m = 442.5
        >>> c = 2
        >>> s = 0.05
        >>> n = 6
        >>> pdf_values = iso.pdf(x=x,loc=m/c,mass=m,charge=c,sigma=s,num_peaks=n)

        The ``.pdf()`` method is then calling the internal ``._pdf()``
        method (as in ``scipy.stats.rv_continuous``), which returns an array of
        pdf evaluations at positions in input array ``x``. Note that ``mass``,
        is solely used for calculation of the weights. To shift the distribution's
        first peak from 0 to the monoisotopic mass-to-charge ratio, ``loc`` must be used.
        However using ``scale`` with this subclass is not supported.
        Use ``charge`` and ``sigma`` instead to tune the shape of the distribution.

    References:

        .. [1] E. J. Breen, F. G. Hopwood, K. L. Williams, and M. R. Wilkins,
               “Automatic poisson peak harvesting for high throughput protein identification,”
               ELECTROPHORESIS: An International Journal, vol. 21, Art. no. 11, 2000,
               doi: 10.1002/1522-2683(20000601)21:11<2243::AID-ELPS2243>3.0.CO;2-K.

    Warning:

        This class is overwriting ``._pdf()`` and ``._rvs()`` as well as
        ``.pdf()`` and ``.rvs()``. While the first two are meant to be customized,
        this is rather not the case for the latter two. However, to allow for
        passing a vector of different standard deviations for each gauss bell a custom
        handling of these two methods was necessary.

    """

    def __init__(self, decoy: bool = False):
        super().__init__()
        self.decoy = decoy

    def pdf(
        self,
        x: ArrayLike,
        mass: ArrayLike,
        charge: ArrayLike,
        sigma: ArrayLike,
        num_peaks: ArrayLike,
        loc: ArrayLike = 0,
        set_peak_height_to_1=False,
    ) -> np.ndarray:
        """Calculates probability density function (PDF)

        Calculates PDF for given position ``x``
        and parameters by evaluating and adding the PDFs of num_peaks
        weighted normal distributions.

        Important: ``mass`` is only for calculation of the components' weights. Variable ``loc``
            is used to set the location of the monoisotopic peak.

        Args:
            x (ArrayLike): Positions to calculate PDF at.
            mass (ArrayLike): Monoisotopic mass of the peptide [Da].
            charge (ArrayLike): Charge of peptide.
            sigma (ArrayLike): Standard deviation(s) of gauss bells.
            num_peaks (ArrayLike) : Number of peaks to consider.
            loc (ArrayLike, optional): Location of monoisotopic peak. Defaults to 0.
            set_peak_height_to_1 (bool, optional): If True, each gauss bells maximum is
                set to its weight.

        Returns:
            ``np.ndarray``. Evaluations of PDF at positions in ``x`` under given parametrization.

        Raises:
            ``AssertionError``, if sigma is not scalar and shape of sigma does not correspond to number of peaks
            ``AssertionError``, if x has more than one dimension.

        Examples:
            For calculation of probability density at position ``x``
            ,with ``loc = 0``:

            >>> iso = IsotopicAveragineDistribution()
            >>> x = np.arange(0,5)/10
            >>> y = iso.pdf(x, mass=301.2, charge=1, sigma=0.05, num_peaks=6)
            >>> print(y)
            [6.88118553e+00 9.31267193e-01 2.30838058e-03 1.04800316e-07
             8.71444728e-14]
        """
        # check and transform args for pdf
        if np.asarray(sigma).ndim < 1:
            sigma = np.repeat(sigma, num_peaks)
        if np.asarray(sigma).size == num_peaks:
            sigma = np.asarray(sigma).reshape((-1, 1))
        else:
            sigma = np.asarray(sigma)
            assert sigma.size == num_peaks * x.size
        x = np.atleast_1d(x)
        assert (
            x.ndim == 1
        ), f"x has shape {x.shape} with {x.ndim} dimensions. x must be 1D maximal"
        mass = np.atleast_1d(mass)
        charge = np.atleast_1d(charge)
        num_peaks = np.atleast_1d(num_peaks)

        # transform x
        _x = x - loc
        # return densities
        return self._pdf(
            _x,
            mass,
            charge,
            sigma,
            num_peaks,
            set_peak_height_to_1=set_peak_height_to_1,
        )

    def _pdf(
        self,
        x: np.ndarray,
        mass: np.ndarray,
        charge: np.ndarray,
        sigma: np.ndarray,
        num_peaks: np.ndarray,
        set_peak_height_to_1: bool,
    ) -> np.ndarray:
        """Calculates probability density function (PDF)

        This method is for internal use only, use ``.pdf()`` instead.
        Overwrites ``scipy.rv_continuous._pdf``. Is internally called by ``.pdf()``
        method. Calculates PDF for given position ``x``
        and parameters by evaluating and adding pdf of num_peaks
        weighted normal distributions.

        Important: ``mass`` is only for calculation of the components' weights. Variable ``loc`` in
        ``.pdf()`` is used to set the location of the monoisotopic peak.


        Args:
            x (np.ndarray): Positions to calculate PDF at.
            mass (np.ndarray): Monoisotopic mass of the peptide [Da].
            charge (np.ndarray): Charge of peptide.
            sigma (np.ndarray): Standard deviation(s) of gauss bells.
            num_peaks (np.ndarray) : Number of peaks to consider.
            set_peak_height_to_1 (bool): If True, each gauss bells maximum is
                set to its weight.

        Returns:
            ``np.ndarray``. Evaluations of PDF at positions in ``x`` under given parametrization.
        """

        # broadcast 1D arrays
        x, m, c, n = np.broadcast_arrays(x, mass, charge, num_peaks)
        # broadcast sigma which is potentially 2D
        _, s = np.broadcast_arrays(x, sigma)
        p = np.zeros_like(x)

        # cache calculated averagine weights
        weights_cache = {}

        for idx in range(x.size):
            p_xi = 0
            x_i, m_i, c_i, n_i = x[idx], m[idx], c[idx], n[idx]
            means = np.arange(n_i) / c_i
            s_i = s[:, idx]
            sigmas = s_i
            # reveresed weights
            if self.decoy:
                if n_i not in weights_cache:
                    weights_cache[n_i] = self.non_averagine_isotopic(m_i, n_i)
                weights = weights_cache[n_i]
            # usual case -> weights via averagine-like model
            else:
                if n_i not in weights_cache:
                    weights_cache[n_i] = self.averagine_isotopic(m_i, n_i)
                weights = weights_cache[n_i]

            for μ, w, σ in zip(means, weights, sigmas):
                if set_peak_height_to_1:
                    p_xi += w * np.exp(-0.5 * ((x_i - μ) / σ) ** 2)
                else:
                    p_xi += (
                        w
                        * 1
                        / (σ * np.sqrt(2 * np.pi))
                        * np.exp(-0.5 * ((x_i - μ) / σ) ** 2)
                    )
            p[idx] = p_xi

        return p

    def _rvs(
        self, mass, charge, sigma, num_peaks, size, random_state
    ) -> np.ndarray[float]:
        """Generation of random variable samples

        Overwrites ``scipy.stats.rv_generic._rvs()``. This sampling method
        first samples a component and then
        samples from the normal distribution of the given component.

        Args:
            mass: Monoisotopic mass of the peptide [Da].
            charge: Charge of peptide.
            sigma: Standard deviation(s) of gauss bells.
            num_peaks: Number of peaks to consider.
            size : Sample Size.
            random_state : e.g. numpy random number generator.

        Returns:
            np.ndarray[float]: Samples from distribution.

        Examples:
            Used via ``.rvs()`` method:
            >>> iso = IsotopicAveragineDistribution()
            >>> y = iso.rvs(loc=301.2, mass = 301.2,charge = 1,sigma=0.05,num_peaks=6,size=5,random_state=np.random.default_rng(2022))
            >>> print(y)
            [301.19520469 301.07622945 301.18164187 301.22961325 301.05343757]
        """
        us = random_state.uniform(size=size)
        devs_std = random_state.normal(size=size)

        if self.decoy:
            weights = self.non_averagine_isotopic(mass, num_peaks).cumsum()
        else:
            weights = self.averagine_isotopic(mass, num_peaks).cumsum()

        def _get_component(u: float, weights_cum_sum: ArrayLike = weights) -> int:
            for idx, weight_sum in enumerate(weights_cum_sum):
                if u < weight_sum:
                    return idx

        comps = np.zeros_like(us)
        devs = np.zeros_like(us)
        for idx, u in enumerate(us):
            comp = _get_component(u)
            comps[idx] = comp
            devs[idx] = sigma[comp] * devs_std[idx]

        values = comps / charge + devs
        return values

    def rvs(self, mass, charge, sigma, num_peaks, loc=0, size=None, random_state=None):
        """Generation of random variable samples

        First samples the component of a sample, than from the normal distribution of
        the sampled component. Component is sampled based on weights
        calculated by either ``.averagine_isotopic()`` or ``.non_averagine_isotopic()``.

        Args:
            mass: Monoisotopic mass of the peptide [Da].
            charge: Charge of peptide.
            sigma: Standard deviation(s) of gauss bells.
            num_peaks: Number of peaks to consider.
            loc: Location of monoisotopic peak.
            size : Sample Size.
            random_state : e.g. numpy random number generator.

        Returns:
            np.ndarray[float]: Samples from distribution.

        Examples:
            Used via ``.rvs()`` method:
            >>> iso = IsotopicAveragineDistribution()
            >>> y = iso.rvs(loc=301.2, mass = 301.2,charge = 1,sigma=0.05,num_peaks=6,size=5,random_state=np.random.default_rng(2022))
            >>> print(y)
            [301.19520469 301.07622945 301.18164187 301.22961325 301.05343757]
        """
        # check arguments for rvs
        if np.asarray(sigma).ndim < 1:
            sigma = np.repeat(sigma, num_peaks)
        elif np.asarray(sigma).size == num_peaks:
            sigma = np.asarray(sigma).reshape((-1, 1))
        else:
            raise ValueError(
                f"Sigma is neither scalar nor does sigma's shape ({np.asarray(sigma).shape}) correspond to num_peaks"
            )
        if size == None:
            size = 1
        if random_state == None:
            random_state = self.random_state

        # return rvs
        return self._rvs(mass, charge, sigma, num_peaks, size, random_state) + loc

    def rvs_to_intensities(
        self,
        loc: float,
        mass: float,
        charge: int,
        sigma: ArrayLike,
        num_peaks: int,
        bin_width: float,
        signal_per_molecule: float = 1,
        detection_limit: float = 0,
        size: int = 1000,
        random_state: np.random.RandomState = None,
        full_return: bool = False,
    ) -> Union[
        tuple[list[list[float]], list[tuple[float]], list[float]],
        tuple[np.ndarray[float], np.ndarray[float]],
    ]:
        """Simulates (m/z, intensity) data.

        The method draws ``size`` samples from ``.rvs()`` and bins theses samples based on ``bin_width``.
        Each sample is considered a single molecule that contributes ``signal_per_molecule`` arbitrary units
        to intensity of its bin.

        Args:
            loc (float): mass-to-charge ratio of monoisotopic peak.
            mass (float): Mass of monoisotopic peak.
            charge (int): Charge of peptide.
            sigma (ArrayLike): Standard deviation(s) of gauss bells.
            num_peaks (int): Number of (isotopic) peaks to consider.
            bin_width (float): Size of bins that are reduced to an intensity.
            signal_per_molecule (float, optional): Number of added arbitrary units to intensity per molecule.
                Defaults to 1.
            detection_limit (float, optional): Minimal intensity that can be detected.
                Defaults to 0.
            size (int, optional): Sample Size. Defaults to 1000.
            random_state (Optional[np.random.RandomState],optional) : e.g. numpy random number generator. Defaults to None.
                If None, random_state is set to ``self.random_state``
            full_return (bool, optional): Wether to return list of bins and list of bin's start and end values.
                Defaults to False.

        Returns:
            Union[tuple[list[list[float]],list[tuple[float]],list[float]],tuple[np.ndarray[float],np.ndarray[float]]]: If ``full_return`` is set to true
                tuple of three lists is returned: ``bins``, ``bins_start_end`` and ``bins_intensities``. ``bins`` is a list of bins that are themselves
                lists with sampled mass-to-charge ratios. ``bins_intensities`` is a list of the calculated intensity of each bin and ``bins_start_end`` stores
                the bins mass-to-charge ratio boundaries as tuple [begin,end).

                If ``full_return`` is set to false (default), a numpy array of the bin's positions (mean of start and end) together with the a numpy array of
                calculated intensities is returned.
        """
        samples_mz = self.rvs(
            loc=loc,
            mass=mass,
            charge=charge,
            sigma=sigma,
            num_peaks=num_peaks,
            size=size,
            random_state=random_state,
        )
        samples_mz.sort()

        bins = []
        s_idx = 0
        current_bin_start = samples_mz[s_idx]
        current_bin_end = current_bin_start + bin_width
        bins_start_end = []
        bins_position = []
        current_bin_intensity = 0
        bins_intensities = []
        current_bin = []

        while s_idx < size:
            s = samples_mz[s_idx]
            while s < current_bin_end:
                # append sample to currently open bin
                current_bin.append(s)
                current_bin_intensity += signal_per_molecule

                # advance
                s_idx += 1
                if s_idx < size:
                    s = samples_mz[s_idx]
                    continue
                else:
                    break
            # store
            if current_bin_intensity >= detection_limit:
                bins.append(current_bin.copy())
                bins_start_end.append((current_bin_start, current_bin_end))
                bins_position.append((current_bin_start + current_bin_end) / 2)
                current_bin.append(s)
                bins_intensities.append(current_bin_intensity)

            else:
                # no recording, intensity too low
                pass

            # reset
            current_bin.clear()
            current_bin_start = s
            current_bin_end = current_bin_start + bin_width
            current_bin_intensity = 0
            current_bin_intensity += signal_per_molecule

            # advance
            s_idx += 1

        if full_return:
            return (bins, bins_start_end, bins_intensities)
        else:
            return (np.array(bins_position), np.array(bins_intensities))

    @staticmethod
    def averagine_isotopic(mass: np.ndarray, num_peaks: int) -> np.ndarray:
        """Calculates weights for isotopic distribution

        Calculates weights for isotopic pattern (gaussian mixture)
        distribution based on averagine-like model ([1]) and normalization.

        Args:
            mass: Monoisotopic mass of peptide.
            num_peaks: Number of considered peaks.
        Returns:
            np.ndarray. Array of weights.
        """
        # averagine approx. Adopted from Hildebrandt Github
        λ = 0.000594 * mass - 0.03091
        n = num_peaks
        iso_w = np.fromiter(
            (np.exp(-λ) * np.power(λ, k) / factorial(k) for k in range(n)), float
        )
        # normalization
        iso_w_norm = iso_w / iso_w.sum()
        return iso_w_norm

    @staticmethod
    def non_averagine_isotopic(mass: np.ndarray, num_peaks: int) -> np.ndarray:
        """Calculates weights for isotopic distribution

        Decoy method. Returns weights of ``averagine_isotopic`` inversed.

        Args:
            mass: Monoisotopic mass of peptide.
            num_peaks: Number of considered peaks.
        Returns:
            np.ndarray. Array of weights.
        """
        iso_w_norm = IsotopicAveragineDistribution.averagine_isotopic(mass, num_peaks)
        return np.flip(iso_w_norm, 0)
