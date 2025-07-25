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
#ifndef sitkImageSeriesReader_h
#define sitkImageSeriesReader_h

#include "sitkMacro.h"
#include "sitkImage.h"
#include "sitkImageReaderBase.h"
#include "sitkMemberFunctionFactory.h"
#include "sitkProcessObjectDeleter.h"

namespace itk::simple
{

/** \class ImageSeriesReader
 * \brief Read series of image files into a SimpleITK image.
 *
 * For some image formats such as DICOM, images also contain
 * associated meta-data (e.g. imaging modality, patient name
 * etc.). By default the reader does not load this information
 *(saves time). To load the meta-data you will need to explicitly
 * configure the reader, MetaDataDictionaryArrayUpdateOn, and
 * possibly specify that you also want to load the private
 * meta-data LoadPrivateTagsOn.
 *
 * Once the image series is read the meta-data is directly
 * accessible from the reader.
 *
 * \note DICOM tags are represented as strings in the meta-data
 * dictionary(s), therefore "0020|000D" and "0020|000d" are
 * different when accessing the tag value. This differs from
 * the hexadecimal numbers they represent, 0020|000D and 0020|000d
 * are equivalent. The ITK meta-data dictionary is string based
 * and uses lower case to represent the hexadecimal number read
 * from disk, so 0020|000d will work as a key and 0020|000D will
 * not be found in the dictionary (results in an exception if
 * attempting to access).
 * It is recommended to use lower case when setting and accessing
 * DICOM tags.
 *
 * \note If the pixel type for the returned image is not specified
 * it is deduced from the first image in the series. This approach
 * is computationally efficient and assumes that all images in a
 * series have the same pixel type. In rare situations this is not
 * the case, not all images have the same pixel type.
 * If this leads to a narrowing conversion (e.g. first image pixel
 * type is unsigned int and others are float) the returned
 * image does not represent the data correctly. To resolve such
 * situations, explicitly specify the expected pixel type via the
 * SetOutputPixelType method before reading the series.
 *
 * \sa itk::simple::ReadImage for the procedural interface
 **/
class SITKIO_EXPORT ImageSeriesReader : public ImageReaderBase
{
public:
  using Self = ImageSeriesReader;

  ~ImageSeriesReader() override;

  ImageSeriesReader();

  /** Print ourselves to string */
  std::string
  ToString() const override;

  /** return user readable name of the filter */
  std::string
  GetName() const override
  {
    return std::string("ImageSeriesReader");
  }


  /**
   * Set/Get whether the meta-data dictionaries for the slices
   * should be read. Default value is false, because of the
   * additional computation time.
   */
  void
  SetMetaDataDictionaryArrayUpdate(bool metaDataDictionaryArrayUpdate)
  {
    this->m_MetaDataDictionaryArrayUpdate = metaDataDictionaryArrayUpdate;
  }
  bool
  GetMetaDataDictionaryArrayUpdate()
  {
    return this->m_MetaDataDictionaryArrayUpdate;
  }


  /** Set the value of MetaDataDictionaryArrayUpdate to true or false respectively. */
  void
  MetaDataDictionaryArrayUpdateOn()
  {
    return this->SetMetaDataDictionaryArrayUpdate(true);
  }
  void
  MetaDataDictionaryArrayUpdateOff()
  {
    return this->SetMetaDataDictionaryArrayUpdate(false);
  }


  /** \brief Generate a sequence of filenames from a directory with a DICOM data set and a series ID.
   *
   * This method generates a sequence of filenames whose filenames
   * point to DICOM files. The data set may contain multiple series.
   * The seriesID string is used to select a specific series.  The
   * ordering of the filenames is based on one of several strategies,
   * which will read all images in the directory ( assuming there is
   * only one study/series ).
   *
   * \param directory         Set the directory that contains the DICOM data set.
   * \param recursive         Recursively parse the input directory.
   * \param seriesID          Set the name that identifies a
   * particular series. Default value is an empty string which
   * will return the file names associated with the first series
   * found in the directory. If the parameter value was obtained
   * from a call to GDCMSeriesIDs, the value of the
   * useSeriesDetails parameter must be the same here and in the
   * call to GDCMSeriesIDs.
   * \param useSeriesDetails  Use additional series information
   * such as ProtocolName and SeriesName to identify when a single
   * SeriesUID contains multiple 3D volumes - as can occur with
   * perfusion and DTI imaging.
   * \param loadSequences     Parse any sequences in the DICOM data set. Loading DICOM files is faster when sequences
   *are not needed.
   *
   * \sa itk::GDCMSeriesFileNames
   **/
  static std::vector<PathType>
  GetGDCMSeriesFileNames(const PathType &    directory,
                         const std::string & seriesID = "",
                         bool                useSeriesDetails = false,
                         bool                recursive = false,
                         bool                loadSequences = false);

  /** \brief Get all the seriesIDs from a DICOM data set
   *
   * \param directory  The directory that contains the DICOM data set
   * \param useSeriesDetails  Use additional series information
   * such as ProtocolName and SeriesName to identify when a single
   * SeriesUID contains multiple 3D volumes - as can occur with
   * perfusion and DTI imaging. The parameter value must match the
   * value used in the call to GDCMSeriesFileNames.
   * \sa itk::GDCMSeriesFileNames
   **/
  static std::vector<std::string>
  GetGDCMSeriesIDs(const PathType & directory, bool useSeriesDetails = false);

  void
  SetFileNames(const std::vector<PathType> & fileNames);
  const std::vector<PathType> &
  GetFileNames() const;

  /** Set the relative threshold for issuing warnings about non-uniform sampling. */
  void
  SetSpacingWarningRelThreshold(double spacingWarningRelThreshold);
  double
  GetSpacingWarningRelThreshold() const;

  /** Set/Get/OnOff whether to force orthogonal direction cosines.
   *
   * Do we want to force orthogonal direction cosines? On by default. Turning it off enables proper reading of DICOM
   * series with gantry tilt.
   */
  void
  SetForceOrthogonalDirection(bool forceOrthogonalDirection);
  bool
  GetForceOrthogonalDirection() const;
  void
  ForceOrthogonalDirectionOn();
  void
  ForceOrthogonalDirectionOff();

  /** Set/Get/OnOff whether to reverse the order of the input files.
   *
   * If true, the order of the input file names will be reversed before reading. Defaults to false.
   */
  void
  SetReverseOrder(bool reverseOrder);
  bool
  GetReverseOrder() const;
  void
  ReverseOrderOn();
  void
  ReverseOrderOff();

  Image
  Execute() override;

  /** \brief Get the meta-data dictionary keys for a slice
   *
   * This is only valid after successful execution of this
   * filter and when MetaDataDictionaryArrayUpdate is true. Each
   * element in the array corresponds to a "slice" or filename
   * read during execution.
   *
   * If the slice index is out of range, an exception will be
   * thrown.
   *
   * Returns a vector of keys to the key/value entries in the
   * file's meta-data dictionary. Iterate through with these keys
   * to get the values.
   **/
  std::vector<std::string>
  GetMetaDataKeys(unsigned int slice) const
  {
    return this->m_pfGetMetaDataKeys(slice);
  }

  /** \brief Query a meta-data dictionary for the existence of a key.
   **/
  bool
  HasMetaDataKey(unsigned int slice, const std::string & key) const
  {
    return this->m_pfHasMetaDataKey(slice, key);
  }

  /** \brief Get the value of a meta-data dictionary entry as a string.
   *
   * If the key is not in the dictionary then an exception is
   * thrown.
   *
   * string types in the dictionary are returned as their native
   * string. Other types are printed to string before returning.
   **/
  std::string
  GetMetaData(unsigned int slice, const std::string & key) const
  {
    return this->m_pfGetMetaData(slice, key);
  }


protected:
  template <class TImageType>
  Image
  ExecuteInternal(itk::ImageIOBase *);

private:
  // function pointer type
  typedef Image (Self::*MemberFunctionType)(itk::ImageIOBase *);

  // friend to get access to executeInternal member
  friend struct detail::MemberFunctionAddressor<MemberFunctionType>;
  std::unique_ptr<detail::MemberFunctionFactory<MemberFunctionType>> m_MemberFactory;


  std::function<std::vector<std::string>(int)>         m_pfGetMetaDataKeys;
  std::function<bool(int, const std::string &)>        m_pfHasMetaDataKey;
  std::function<std::string(int, const std::string &)> m_pfGetMetaData;

  // Holder of a process object for active measurements
  std::unique_ptr<itk::ProcessObject, ProcessObjectDeleter> m_Filter;


  std::vector<PathType> m_FileNames;

  double m_SpacingWarningRelThreshold{ 1e-4 };

  bool m_MetaDataDictionaryArrayUpdate{ false };

  bool m_ForceOrthogonalDirection{ true };

  bool m_ReverseOrder{ false };
};

/**
 * \brief ReadImage is a procedural interface to the ImageSeriesReader
 *     class which is convenient for most image reading tasks.
 *
 *  \param fileNames a vector of file names
 *  \param outputPixelType see ImageReaderBase::SetOutputPixelType
 *  \param imageIO see ImageReaderBase::SetImageIO
 *
 *  \note When reading a series of images that have meta-data
 *  associated with them (e.g. a DICOM series) the resulting
 *  image will have an empty meta-data dictionary.
 *  If you need the meta-data dictionaries associated with each
 *  slice then you should use the ImageSeriesReader class.
 *
 *  \note If the pixel type for the returned image is not specified
 *  it is deduced from the first image in the series. This approach
 *  is computationally efficient and assumes that all images in a
 *  series have the same pixel type. If this is not the case,
 *  explicitly specify the widest pixel type in the series as the
 *  outputPixelType.
 *
 * \sa itk::simple::ImageFileReader for reading a single file.
 * \sa itk::simple::ImageSeriesReader for reading a series and meta-data dictionaries.
 */
SITKIO_EXPORT Image
ReadImage(const std::vector<PathType> & fileNames,
          PixelIDValueEnum              outputPixelType = sitkUnknown,
          const std::string &           imageIO = "");
} // namespace itk::simple

#endif
