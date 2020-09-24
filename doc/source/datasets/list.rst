=============
Datasets List
=============


General Description of a Dataset
================================

Dataset Composition
-------------------

A Benzina dataset is, in essence, an indexing over a concatenation of inputs,
targets and possibly filenames with indexing

Dataset Structure
-----------------

A Benzina dataset is structured using the mp4 format

:ftyp: Defines the compatibilities of the mp4 container
:mdat: Concatenation in 2-3 blocks of the inputs, targets and possibly filenames
:moov: Contains the metadata needed to load and present the raw data of *mdat*

       :mvhd: Defines the *timescale* and the *duration* of the container

              :timescale: How many units elapse in 1 second
              :duration: Duration of the container in *timescale* units
              :next_track_id: The id of the next track that could be appended to *moov*

       :trak: * *Benzina input samples track*: This is the first track and it references
                                               all the input samples
              * *Benzina target track*
              * *Benzina filename track*: This track is optional
              * *Video track*: This track is optional. If present it should be
                               positioned last

              Each track can have a *train*, *validation* and *test* variants to
              reference the sets

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: Defines if the track should be displayed
                     :width: Width of the video
                     :height: Height of the video

              :mdia: Contains definitions related to the media type of the data

                     :mdhd: Redefines the *timescale* and the *duration* for the track

                            :timescale: How many units elapse in 1 second
                            :duration: Duration of the track in *timescale* units

                     :hdlr: Defines the media type of the track

                            :handler_type: Defines the type of handler that should
                                           be used to decode the data referenced by the track
                            :name: Human readable name for the track type
                                   (used for debugging)

                     :minf: Defines the characteristics of the media in the track

                            :stbl: Defines the data indexing of the media samples
                                   in the track along with coding information, if
                                   needed, to decode them

                                   :stsd: Provides the information needed to decode
                                          the media samples
                                   :stts: Defines the mapping from decoding time
                                          to sample number

                                          :sample_count: The number of samples in
                                                         the track
                                          :sample_delta: The interval in ``timescale``
                                                         units for which a new sample
                                                         should be decoded

                                   :stsz: Defines the size of each samples

                                          :sample_count: Number of samples in the
                                                         track
                                          :entry_size: Size of the sample. This field
                                                       is repeated for each sample

                                   :stsc: Defines the chunks splitting the data
                                   :stco: Defines the chunks offset

                                          :entry_count: Number of chunks
                                          :chunk_offset: The chunk offset. This field
                                                         is repeated for each chunk

Dataset's Input Sample Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Benzina dataset's input sample can also be structured using the mp4 format.
It is roughly the same as the dataset's structure with the differences that
*mdat* will contains the raw concatenation of a single input, its target,
possibly filename and possibly a 512 x 512 thumbnails stream.

.. _imagenet_2012:

ImageNet 2012
=============

`ImageNet 2012 <http://image-net.org/>`_ classification dataset. It contains
two size of the images along with their classification target and filename:

* Resized high resolution images each with a smaller edge of at most 512 while
  preserving the aspect ratio. This set is accessed by referencing the
  *bzna_input* track of the input samples.
* Resized images each  with a longer edge of at most 512 while preserving the
  aspect ratio. This set is accessed by referencing the *bzna_thumb* track of
  the input samples.

The dataset is represented by :py:class:`~benzina.dataset.ImageNet` which
simplifies the iteration of the data as a classification dataset.

.. warning::
   81 images are currently missing from the dataset and 111 had to be first
   transcoded to PNG prior to the final H.265 format. More details can be found
   in the dataset's README.

.. warning::
   High resolution images stored in the the *bzna_input* track of the input
   samples are currently not available through the
   :py:class:`~benzina.torch.dataloader.DataLoader`. Their widely varying sizes
   prevent them from being decoded using a single hardware decoder
   configuration. The selected solution is to represent the images in the HEIF
   format which will be completed in future development.

Dataset Composition
-------------------

The dataset is composed of a train set, followed by a validation set then a
test set for a total of 1 431 167 entries. Targets and filenames are provided
for each sets:

* | **Train set**
  | Entries 1 to 1281167 (1 281 167 entries)
* | **Validation set**
  | Entries 1281168 to 1331167 (50 000 entries)
* | **Test set**
  | Entries 1331168 to 1431167 (100 000 entries)

Dataset Structure
-----------------

ilsvrc2012.bzna
^^^^^^^^^^^^^^^

:ftyp: Defines the compatibilities of the mp4 container

       :major_brand: isom
       :minor_version: 0
       :compatible_brands: bzna, isom

:mdat: Raw concatenation in 3 blocks of the images, targets and filenames

       * Concatenation of .mp4 files containing a single image, a thumbnail of a
         maximum size of 512 x 512 if the image does not already fit this resolution,
         the image's original filename and the target associated with the image
       * Concatenation of images' targets as little-endian int64
       * Concatenation of images' original filename

:moov: Contains the metadata needed to load and present the raw data of *mdat*

       :mvhd: Defines the *timescale* and the *duration* of the container

              :timescale: 20
              :duration: 20 * 1 431 167
              :next_track_id: The id of the next track that could be appended to *moov*

       :trak: *Benzina input samples track*

              This track references all the images of the dataset

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: 000000 -- This value informs that the track is not
                                       for display purpose
                     :width: 0.0 -- This value reflects the variance in size of the frames
                     :height: 0.0 -- This value reflects the variance in size of the frames

              :mdia: Contains definitions related to the media type of the data

                     :mdhd: Redefines the *timescale* and the *duration* for the track

                            :timescale: 20
                            :duration: 20 * 1 431 167

                     :hdlr: Defines the media type of the track

                            :handler_type: ``meta``
                            :name: ``bzna_input``

                     :minf: Defines the characteristics of the media in the track

                            :nmhd: No specific media header is identified for the track

                            :stbl: Defines the data indexing of the media samples
                                   in the track along with coding information, if
                                   needed, to decode them

                                   :stsd: Provides the information needed to decode
                                          the media samples

                                          :mett: Defines the metadata as being text based

                                                 :mime_format: ``application/octet-stream``

                                   :stts: Defines the mapping from decoding time
                                          to sample number

                                          :sample_count: 1 431 167
                                          :sample_delta: 20

                                   :stsz: Defines the size of each samples

                                          :sample_count: 1 431 167
                                          :entry_size: Size of the sample. This field
                                                       is repeated for each sample

                                   :stsc: Defines the chunks splitting the data

                                          :first_chunk: 1
                                          :samples_per_chunk: 1
                                          :sample_description_index: 1

                                          This definition means to consider that
                                          all samples are contained in their own chunk

                                   :stco: Defines the chunks offset

                                          :entry_count: 1 431 167
                                          :chunk_offset: The chunk offset. This field
                                                         is repeated for each chunk,
                                                         i.e. for each sample

       :trak: *Benzina target track*

              This track is roughly the same as the *Benzina input track* with the
              following differences

              :mdia: Contains definitions related to the media type of the data

                     :hdlr: Defines the media type of the track

                            :handler_type: ``meta``
                            :name: ``bzna_target``

       :trak: *Benzina filename track*

              This track is roughly the same as the *Benzina input track* with the
              following differences

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: 000003 -- This value informs that the track is enabled
                                       and can be used in the presentation
                     :width: 0.0 -- This value informs that no width has be predefined
                                    for this track
                     :height: 0.0 -- This value informs that no height has be predefined
                                     for this track

              :mdia: Contains definitions related to the media type of the data

                     :hdlr: Defines the media type of the track

                            :handler_type: ``meta``
                            :name: ``bzna_fname``

                     :minf: Defines the characteristics of the media in the track

                            :stbl: Defines the data indexing of the media samples
                                   in the track along with coding information, if
                                   needed, to decode them

                                   :stsd: Provides the information needed to decode
                                          the media samples

                                          :mett: Defines the metadata as being text based

                                                 :mime_format: ``text/plain``

       :trak: *Video track*

              This track allows to play the thumbnails of the dataset's frames

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: 000003 -- This value informs that the track is enabled
                                       and can be used in the presentation
                     :width: 512.0
                     :height: 512.0

              :mdia: Contains definitions related to the media type of the data

                     :mdhd: Redefines the *timescale* and the *duration* for the track

                            :timescale: 20
                            :duration: 1 431 167

                     :hdlr: Defines the media type of the track

                            :handler_type: ``vide``
                            :name: ``VideoHandler``

                     :minf: Defines the characteristics of the media in the track

                            :vmhd: Video media header is identified for the track

                            :stbl: Defines the data indexing of the media samples
                                   in the track along with coding information, if
                                   needed, to decode them

                                   :stsd: Provides the information needed to decode
                                          the media samples

                                          :avc1: Defines the AVC coding information

                                                 :width: 512
                                                 :height: 512
                                                 :horizresolution: 72
                                                 :horizresolution: 72

                                   :stts: Defines the mapping from decoding time
                                          to sample number

                                          :sample_count: 1 431 167
                                          :sample_delta: 1

                                   :stsz: Defines the size of each samples

                                          :sample_count: 1 431 167
                                          :entry_size: Size of the sample. This field
                                                       is repeated for each sample

                                   :stsc: Defines the chunks splitting the data

                                          :first_chunk: 1
                                          :samples_per_chunk: 1
                                          :sample_description_index: 1

                                          This definition means to consider that
                                          all samples are contained in their own chunk

                                   :stco: Defines the chunks offset

                                          :entry_count: 1 431 167
                                          :chunk_offset: The chunk offset. This field
                                                         is repeated for each chunk,
                                                         i.e. for each sample

Dataset's Input Samples Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Benzina ImageNet dataset's input sample is structured using the mp4 format.

:ftyp: Defines the compatibilities of the mp4 container

       :major_brand: isom
       :minor_version: 0
       :compatible_brands: bzna, isom

:mdat: Raw concatenation of the image, thumbnail, target and filename:

       * A single image in H.265 format. The image is put in a frame with a size
         of a product of 512 in the 2 dimensions. The padding to make the image
         fit is filled with a smear of the image's borders
       * A thumbnail in H.265 format. The image is put in a frame of size 512 x 512.
         The image is first resized to have its longest side be of 512. The padding
         to make the thumbnail fit the frame is filled with a smear of the image's
         borders. There will be no explicit thumbnail if the image already fit the
         thumbnail's frame
       * The image's target in a little-endian int64
       * The image's original filename

:moov: Contains the metadata needed to load and present the raw data of *mdat*

       :mvhd: Defines the *timescale* and the *duration* of the container

              :timescale: 20
              :duration: 20
              :next_track_id: The id of the next track that could be appended to *moov*

       :trak: *Benzina input track*

              This track references an image

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: 000000 -- This value informs that the track is not
                                       for display purpose
                     :width: Width of the image without padding
                     :height: Height of the image without padding

              :mdia: Contains definitions related to the media type of the data

                     :mdhd: Redefines the *timescale* and the *duration* for the track

                            :timescale: 20
                            :duration: 20

                     :hdlr: Defines the media type of the track

                            :handler_type: ``vide``
                            :name: ``bzna_input``

                     :minf: Defines the characteristics of the media in the track

                            :vmhd: Video media header is identified for the track
                            :stbl: Defines the data indexing of the media samples
                                   in the track along with coding information, if
                                   needed, to decode them

                                   :stsd: Provides the information needed to decode
                                          the media samples

                                          :avc1: Defines the AVC coding information

                                                 :width: Width of the image's frame.
                                                         This is a product of 512
                                                 :height: Height of the image's frame.
                                                          This is a product of 512
                                                 :horizresolution: 72
                                                 :horizresolution: 72

                                                 :clap: Defines the clean aperture
                                                        of the image to remove the
                                                        padding

                                                        :clean_aperture_width_n: Width of the image without padding
                                                        :clean_aperture_width_d: 1
                                                        :clean_aperture_height_n: Height of the image without padding
                                                        :clean_aperture_height_d: 1
                                                        :horiz_off_n: The negative value of the width's padding
                                                        :horiz_off_d: 2
                                                        :vert_off_n: The negative value of the height's padding
                                                        :vert_off_d: 2

                                   :stts: Defines the mapping from decoding time
                                          to sample number

                                          :sample_count: 1
                                          :sample_delta: 20

                                   :stsz: Defines the size of each samples

                                          :sample_count: 1
                                          :entry_size: Size of the input

                                   :stsc: Defines the chunks splitting the data

                                          :first_chunk: 1
                                          :samples_per_chunk: 1
                                          :sample_description_index: 1

                                   :stco: Defines the chunks offset

                                          :entry_count: 1
                                          :chunk_offset: The chunk offset

       :trak: *Benzina thumbnail track*

              This track references an image's thumbnail. If the image already fits
              a thumbnail's frame, then this track will reference the same data as
              in the *Benzina input track*. In any case, it is roughly the same as
              the *Benzina input track* with the following differences

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: 000003 -- This value informs that the track is enabled
                                       and can be used in the presentation
                     :width: Width of the thumbnail without padding
                     :height: Height of the thumbnail without padding

              :mdia: Contains definitions related to the media type of the data

                     :hdlr: Defines the media type of the track

                            :handler_type: ``vide``
                            :name: ``bzna_thumb``

       :trak: *Benzina target track*

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: 000000 -- This value informs that the track is not
                                       for display purpose
                     :width: 0.0 -- This value informs that the width has not been
                                    predefined for this track
                     :height: 0.0 -- This value informs that no height has not been
                                     predefined for this track

              :mdia: Contains definitions related to the media type of the data

                     :mdhd: Redefines the *timescale* and the *duration* for the track

                            :timescale: 20
                            :duration: 20

                     :hdlr: Defines the media type of the track

                            :handler_type: ``meta``
                            :name: ``bzna_target``

                     :minf: Defines the characteristics of the media in the track

                            :nmhd: No specific media header is identified for the track
                            :stbl: Defines the data indexing of the media samples
                                   in the track along with coding information, if
                                   needed, to decode them

                                   :stsd: Provides the information needed to decode
                                          the media samples

                                          :mett: Defines the metadata as being text based

                                                 :mime_format: ``application/octet-stream``

       :trak: *Benzina filename track*

              This track is roughly the same as the *Benzina target track* with the
              following differences

              :tkhd: Defines the resolution of the video and if the track should
                     be displayed by an mp4 player

                     :flags: 000003 -- This value informs that the track is enabled
                                       and can be used in the presentation
                     :width: 0.0 -- This value informs that no width has be predefined
                                    for this track
                     :height: 0.0 -- This value informs that no height has be predefined
                                     for this track

              :mdia: Contains definitions related to the media type of the data

                     :hdlr: Defines the media type of the track

                            :handler_type: ``meta``
                            :name: ``bzna_fname``

                     :minf: Defines the characteristics of the media in the track

                            :stbl: Defines the data indexing of the media samples
                                   in the track along with coding information, if
                                   needed, to decode them

                                   :stsd: Provides the information needed to decode
                                          the media samples

                                          :mett: Defines the metadata as being text based

                                                 :mime_format: ``text/plain``
