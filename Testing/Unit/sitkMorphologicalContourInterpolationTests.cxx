/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include <SimpleITKTestHarness.h>
#include <sitkMorphologicalContourInterpolatorImageFilter.h>
#include <sitkImage.h>
#include <sitkPasteImageFilter.h>
#include <sitkHashImageFilter.h>

namespace
{

itk::simple::Image
MakeSparseLabelVolume()
{
  namespace sitk = itk::simple;

  sitk::Image volume(32, 32, 16, sitk::sitkUInt8);

  sitk::Image slice(32, 32, 1, sitk::sitkUInt8);
  for (unsigned int y = 10; y < 22; ++y)
  {
    for (unsigned int x = 10; x < 22; ++x)
    {
      slice.SetPixelAsUInt8({ x, y, 0 }, 1);
    }
  }
  volume = sitk::Paste(volume, slice, slice.GetSize(), { 0, 0, 0 }, { 0, 0, 3 });
  volume = sitk::Paste(volume, slice, slice.GetSize(), { 0, 0, 0 }, { 0, 0, 11 });
  return volume;
}

} // namespace

TEST(BasicFilters, MorphologicalContourInterpolatorImageFilter_Defaults)
{
  namespace sitk = itk::simple;

  const sitk::Image input = MakeSparseLabelVolume();

  sitk::MorphologicalContourInterpolatorImageFilter filter;
  EXPECT_EQ(0.0, filter.GetLabel());
  EXPECT_EQ(-1, filter.GetAxis());
  EXPECT_TRUE(filter.GetHeuristicAlignment());
  EXPECT_TRUE(filter.GetUseDistanceTransform());
  EXPECT_FALSE(filter.GetUseBallStructuringElement());
  EXPECT_TRUE(filter.GetUseExtrapolation());

  const sitk::Image output = filter.Execute(input);
  EXPECT_EQ(input.GetSize(), output.GetSize());
  EXPECT_EQ(sitk::sitkUInt8, output.GetPixelID());

  EXPECT_EQ(0, output.GetPixelAsUInt8({ 16, 16, 2 }));
  EXPECT_EQ(0, output.GetPixelAsUInt8({ 16, 16, 12 }));
  for (unsigned int z = 3; z <= 11; ++z)
  {
    EXPECT_EQ(1, output.GetPixelAsUInt8({ 16, 16, z })) << "slice " << z;
  }
  EXPECT_EQ(0, output.GetPixelAsUInt8({ 2, 2, 7 }));
}

TEST(BasicFilters, MorphologicalContourInterpolatorImageFilter_SingleAxis)
{
  namespace sitk = itk::simple;

  const sitk::Image input = MakeSparseLabelVolume();

  sitk::MorphologicalContourInterpolatorImageFilter filter;
  filter.SetAxis(0);
  const sitk::Image alongX = filter.Execute(input);
  EXPECT_EQ(sitk::Hash(input), sitk::Hash(alongX));

  filter.SetAxis(2);
  const sitk::Image alongZ = filter.Execute(input);
  EXPECT_EQ(1, alongZ.GetPixelAsUInt8({ 16, 16, 7 }));

  EXPECT_EQ(sitk::Hash(alongZ), sitk::Hash(sitk::MorphologicalContourInterpolator(input)));
  EXPECT_EQ(sitk::Hash(alongZ), sitk::Hash(sitk::MorphologicalContourInterpolator(input, 0.0, 2)));
}

TEST(BasicFilters, MorphologicalContourInterpolatorImageFilter_2DThrows)
{
  namespace sitk = itk::simple;

  const sitk::Image                                 input(32, 32, sitk::sitkUInt8);
  sitk::MorphologicalContourInterpolatorImageFilter filter;
  EXPECT_ANY_THROW(filter.Execute(input));
}
