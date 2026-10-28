# ==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==========================================================================*/

import pytest

import SimpleITK as sitk
import sys


def assert_image_almost_equal(img1, img2, max_difference=1e-8, msg=None):
    """Compare with the maximum absolute difference between two images"""

    minMaxFilter = sitk.StatisticsImageFilter()
    minMaxFilter.Execute(sitk.Abs(img1 - img2))

    if minMaxFilter.GetMaximum() > max_difference:
        print(
            "min: ",
            minMaxFilter.GetMinimum(),
            " max: ",
            minMaxFilter.GetMaximum(),
            " mean: ",
            minMaxFilter.GetMean(),
        )
        pytest.fail(msg or f"Images differ by more than {max_difference}")


def test_procedure_resample():
    """Test the custom Resample procedure."""

    img = sitk.GaborSource(sitk.sitkFloat32, [64, 64], frequency=0.05)

    tx = sitk.Euler2DTransform([32, 32], 0.1, [2, -3])

    filter = sitk.ResampleImageFilter()
    filter.SetTransform(tx)
    filter.SetOutputPixelType(sitk.sitkFloat64)
    filter.SetDefaultPixelValue(255)
    filter.SetReferenceImage(img)
    baseline_image = filter.Execute(img)

    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(img, tx, sitk.sitkLinear, 255.0, sitk.sitkFloat64, False),
    )
    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(
            img, img, tx, sitk.sitkLinear, 255.0, sitk.sitkFloat64, False
        ),
    )
    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(
            img,
            (64, 64),
            tx,
            sitk.sitkLinear,
            (0.0, 0.0),
            (1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
            255.0,
            sitk.sitkFloat64,
            False,
        ),
    )

    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(
            img,
            transform=tx,
            defaultPixelValue=255.0,
            outputPixelType=sitk.sitkFloat64,
        ),
    )
    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(
            img,
            img,
            transform=tx,
            defaultPixelValue=255.0,
            outputPixelType=sitk.sitkFloat64,
        ),
    )
    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(
            img,
            [64, 64],
            transform=tx,
            defaultPixelValue=255.0,
            outputPixelType=sitk.sitkFloat64,
        ),
    )

    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(
            img,
            referenceImage=img,
            transform=tx,
            defaultPixelValue=255.0,
            outputPixelType=sitk.sitkFloat64,
        ),
    )
    assert_image_almost_equal(
        baseline_image,
        sitk.Resample(
            img,
            size=(64, 64),
            transform=tx,
            defaultPixelValue=255.0,
            outputPixelType=sitk.sitkFloat64,
        ),
    )

    with pytest.raises(TypeError) as excinfo:
        sitk.Resample(img, img, wrong_keyword1=img)
    assert "unexpected keyword argument" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        sitk.Resample(img, [64, 64], wrong_keyword2=img)
    assert "unexpected keyword argument" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        sitk.Resample(img, referenceImage=img, wrong_keyword3=img)
    assert "unexpected keyword argument" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        sitk.Resample(img, size=[64, 64], wrong_keyword4=img)
    assert "unexpected keyword argument" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        sitk.Resample(img, tx, wrong_keyword5=img)
    assert "unexpected keyword argument" in str(excinfo.value)

def test_paste_filter():
    """Test the paste filter"""
    img1 = sitk.GaborSource(sitk.sitkFloat32, [64, 64, 64], frequency=0.05)
    img2 = sitk.Image([32, 32, 32], sitk.sitkFloat32)

    # Just verify this runs without error
    sitk.Paste(img1, img2)


def test_landmark_based_transform_initializer():
    """
    Test downcast of LandmarkBasedTransformInitializer return Transform
    """
    point_set = [i for i in range(12)]

    tx_initializer = sitk.LandmarkBasedTransformInitializerFilter()
    tx_initializer.SetFixedLandmarks(point_set)
    tx_initializer.SetMovingLandmarks(point_set)

    assert tx_initializer.Execute(sitk.Euler2DTransform()).__class__ == sitk.Euler2DTransform
    assert tx_initializer.Execute(sitk.Similarity2DTransform()).__class__ == sitk.Similarity2DTransform
    assert tx_initializer.Execute(sitk.AffineTransform(2)).__class__ == sitk.AffineTransform

    assert tx_initializer.Execute(sitk.VersorRigid3DTransform()).__class__ == sitk.VersorRigid3DTransform
    assert tx_initializer.Execute(sitk.Similarity3DTransform()).__class__ == sitk.Similarity3DTransform
    assert tx_initializer.Execute(sitk.ScaleVersor3DTransform()).__class__ == sitk.ScaleVersor3DTransform
    assert tx_initializer.Execute(sitk.AffineTransform(3)).__class__ == sitk.AffineTransform

@pytest.fixture(scope="module")
def test_images():
    """Fixture providing fixed and moving images for transform initialization"""
    fixed = sitk.Image([16, 16, 16], sitk.sitkFloat32)
    fixed.SetOrigin([32, 32, 32])
    moving = sitk.Image([16, 16, 16], sitk.sitkFloat32)
    return fixed, moving


@pytest.mark.parametrize("transform_name,parameters", [
    ("AffineTransform", (3,)),
    ("Euler3DTransform", ()),
    ("ScaleSkewVersor3DTransform", ()),
    ("ScaleTransform", (3,)),
    ("ScaleVersor3DTransform", ()),
    ("Similarity3DTransform", ()),
    ("VersorRigid3DTransform", ()),
    ("VersorTransform", ()),
])
def test_centered_transform_initializer(test_images, transform_name, parameters):
    """Test CenteredTransformInitializer to check that return Transform is downcasted

    Args:
        test_images: Tuple of (fixed, moving) test images
        transform_name: Name of the transform class to test
        parameters: Parameters for transform constructor
    """
    fixed, moving = test_images
    tx = getattr(sitk, transform_name)(*parameters)

    # Test using the filter
    tx_initializer = sitk.CenteredTransformInitializerFilter()
    tx_initializer.SetOperationMode(sitk.CenteredTransformInitializerFilter.GEOMETRY)
    result_tx = tx_initializer.Execute(fixed, moving, tx)
    assert result_tx.__class__ == tx.__class__, \
        f"Transform class changed for filter execution with {transform_name}"

    # Test using the procedural interface
    result_tx = sitk.CenteredTransformInitializer(
        fixed, moving, tx, sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    assert result_tx.__class__ == tx.__class__, \
        f"Transform class changed for procedural execution with {transform_name}"

def test_minimum_maximum():
    """
    Test the manual written MinimumMaximum procedural method
    """
    img = sitk.Image([32, 32], sitk.sitkFloat32)

    assert sitk.MinimumMaximum(img) == (0.0, 0.0)

    img[0, 0] = -1.0
    img[10, 10] = 2.0
    assert sitk.MinimumMaximum(img) == (-1.0, 2.0)

    img = sitk.Image([32, 32], sitk.sitkUInt8)
    assert sitk.MinimumMaximum(img) == (0.0, 0.0)

    img += 1
    img[10, 10] = 255
    assert sitk.MinimumMaximum(img) == (1.0, 255.0)


def _sparse_label_volume():
    volume = sitk.Image([32, 32, 16], sitk.sitkUInt8)
    square = sitk.Image([12, 12, 1], sitk.sitkUInt8) + 1
    volume = sitk.Paste(volume, square, square.GetSize(), [0, 0, 0], [10, 10, 3])
    volume = sitk.Paste(volume, square, square.GetSize(), [0, 0, 0], [10, 10, 11])
    return volume


def test_fast_bilateral_image_filter():
    """Test the FastBilateralImageFilter wrapped from the ITK FastBilateral module."""
    if not hasattr(sitk, "FastBilateralImageFilter"):
        pytest.skip("FastBilateralImageFilter not available")

    clean = sitk.Image([64, 64], sitk.sitkFloat32)
    half = sitk.Image([32, 64], sitk.sitkFloat32) + 100.0
    clean = sitk.Paste(clean, half, half.GetSize(), [0, 0], [32, 0])
    noisy = sitk.AdditiveGaussianNoise(clean, standardDeviation=5.0, seed=1)

    filter = sitk.FastBilateralImageFilter()
    assert filter.GetDomainSigma() == (4.0, 4.0, 4.0)
    assert filter.GetRangeSigma() == 50.0
    filter.SetDomainSigma(4.0)
    filter.SetRangeSigma(20.0)
    out = filter.Execute(noisy)

    assert out.GetSize() == noisy.GetSize()
    assert out.GetPixelID() == noisy.GetPixelID()

    stats_in = sitk.StatisticsImageFilter()
    stats_in.Execute(noisy - clean)
    stats_out = sitk.StatisticsImageFilter()
    stats_out.Execute(out - clean)
    assert stats_out.GetSigma() < 0.5 * stats_in.GetSigma()
    assert abs(out[31, 32]) < 5.0
    assert out[32, 32] > 95.0

    assert sitk.Hash(out) == sitk.Hash(sitk.FastBilateral(noisy, [4.0, 4.0], 20.0))


def test_variance_image_filter():
    """Test the VarianceImageFilter wrapped from the ITK AdaptiveDenoising module."""
    if not hasattr(sitk, "VarianceImageFilter"):
        pytest.skip("VarianceImageFilter not available")

    img = sitk.Image([8, 8], sitk.sitkFloat32)
    img[3, 3] = 9.0

    filter = sitk.VarianceImageFilter()
    assert filter.GetRadius() == (1, 1, 1)
    out = filter.Execute(img)

    assert out.GetSize() == img.GetSize()
    assert out.GetPixelID() == img.GetPixelID()
    assert out[0, 0] == 0.0
    assert out[3, 3] == pytest.approx(9.0)
    assert out[2, 2] == pytest.approx(9.0)
    assert out[6, 6] == 0.0

    assert sitk.Hash(out) == sitk.Hash(sitk.Variance(img, [1, 1]))


def test_morphological_contour_interpolator_image_filter():
    """Test the MorphologicalContourInterpolatorImageFilter wrapped from the ITK MorphologicalContourInterpolation module."""
    if not hasattr(sitk, "MorphologicalContourInterpolatorImageFilter"):
        pytest.skip("MorphologicalContourInterpolatorImageFilter not available")

    volume = _sparse_label_volume()

    filter = sitk.MorphologicalContourInterpolatorImageFilter()
    assert filter.GetLabel() == 0.0
    assert filter.GetAxis() == -1
    assert filter.GetHeuristicAlignment()
    assert filter.GetUseDistanceTransform()
    assert not filter.GetUseBallStructuringElement()
    assert filter.GetUseExtrapolation()
    out = filter.Execute(volume)

    assert out.GetSize() == volume.GetSize()
    assert out.GetPixelID() == sitk.sitkUInt8
    assert out[16, 16, 2] == 0
    assert out[16, 16, 12] == 0
    for z in range(3, 12):
        assert out[16, 16, z] == 1
    assert out[2, 2, 7] == 0

    assert sitk.Hash(out) == sitk.Hash(sitk.MorphologicalContourInterpolator(volume))

    with pytest.raises(RuntimeError):
        filter.Execute(sitk.Image([32, 32], sitk.sitkUInt8))
