[app]
title = Gemini Cleaner
package.name = geminicleaner
package.domain = org.local

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,txt
source.exclude_dirs = .github, __pycache__, bin, .buildozer, tests

version = 1.0

requirements = python3,kivy,numpy
orientation = portrait
fullscreen = 0

android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_VIDEO
android.api = 30
android.minapi = 24
android.ndk_api = 24
android.archs = arm64-v8a
android.legacy_external_storage = True
android.accept_sdk_license = True
android.allow_backup = True

p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 1