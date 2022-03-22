""" Top-level package for DATGAN """

__author__ = """Gael Lederrey"""
__email__ = 'gael.lederrey@epfl.ch'
__version__ = '2.1.1'

from datgan.datgan import DATGAN
from datgan.evaluation.statistical_assessments import stats_assessment
from datgan.evaluation.ml_efficacy import ml_assessment, transform_results

__all__ = {
    'DATGAN',
    'stats_assessment',
    'ml_assessment',
    'transform_results'
}
