# Deployment Notes

## Instrument Configuration

- Omit the `instruments` key or set it to `null`/`None` to use the default instrument basket.
- Provide an empty list (`[]`) to pause trading entirely.
- Set `merge_default_instruments` to `true` to append the default instruments after any custom subset, including when the subset is empty.
