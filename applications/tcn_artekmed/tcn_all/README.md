# tcn_all

Build-only aggregate application for the TCN operators.

## Purpose

`tcn_all` is a placeholder application. It does not provide a runnable pipeline or
process data. Its CMake target links against every `tcn_*` operator target so that
building this application also compiles all TCN operators and their transitive
dependencies.

This gives the TCN operator collection one build entry point and provides an early
check that the complete operator set can be configured and compiled together.

## Included operators

The aggregate covers the operators in `operators/tcn_artekmed/`:

```text
tcn_cdr_decoder
tcn_cdr_serde
tcn_convert_bgra_to_rgba
tcn_dataset_replayer
tcn_depth_anything
tcn_depthimage_apply_mask
tcn_depthimage_backprojection
tcn_depthimage_fgbg_mask
tcn_depthimage_max_distance
tcn_depthimage_temporal_filter
tcn_depthimage_weights
tcn_device_context
tcn_flatten_tensor
tcn_instance_stats
tcn_label_sampler
tcn_labeled_pointcloud
tcn_langsam
tcn_object_tracking
tcn_panoptic_map
tcn_processing
tcn_shm_io
tcn_shm_serde
tcn_shm_subscriber
tcn_shm_zenoh_sender
tcn_slang_renderer
tcn_stream_merger
tcn_stream_splitter
tcn_stream_synchronizer
tcn_texture_sampler
tcn_ui
tcn_util
tcn_zenoh_publisher
tcn_zenoh_receiver
tcn_zenoh_subscriber
```

The list is intentionally kept aligned with the operator directories. Operators
may still be skipped by CMake when an optional external dependency is unavailable;
see the corresponding operator README and build output for the required
dependencies.

## Building

Build the aggregate application with the usual Holohub command:

```bash
./holohub build tcn_all
```

To run the placeholder application, first start the configured container. Then run
the application from inside the container:

```bash
./holohub run --local --cuda 13 tcn_all
```

It loads the aggregate Python application and prints a success message when the
operator imports complete.
