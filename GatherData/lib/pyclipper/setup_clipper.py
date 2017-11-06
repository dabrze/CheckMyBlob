from distutils import util, log
from distutils.errors import DistutilsSetupError
from distutils import sysconfig
from distutils.command.build_scripts import build_scripts as _build_scripts
import sys, os, re, glob

try:
    from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext
    has_cython = False


class build_clipper_ext(build_ext):
    def build_extensions(self):
        self.compiler.find_library_file = self._find_library_file
        self.find_libraries()
        compiler = os.getenv('CC')
        args = {}
        # unfortunately, distutils doesn't let us provide separate C and C++
        # compilers
        if compiler is not None:
            (ccshared,cflags) = sysconfig.get_config_vars('CCSHARED','CFLAGS')
            args['compiler_so'] = compiler + ' ' + ccshared + ' ' + cflags
        self.compiler.set_executables(**args)

        build_ext.build_extensions(self)
    def _find_library_file(self, dirs, lib, debug=0):
        # Ok if lib has already form of lib*.* then skip generating name
        if re.match("lib.*\.(?:dylib|so|a)", lib):
            shared_f = dylib_f = static_f = lib
        else:
            # Copied from distutils.unixcompiler
            shared_f = self.compiler.library_filename(lib, lib_type='shared')
            dylib_f = self.compiler.library_filename(lib, lib_type='dylib')
            static_f = self.compiler.library_filename(lib, lib_type='static')

        if sys.platform == 'darwin':
            # On OSX users can specify an alternate SDK using
            # '-isysroot', calculate the SDK root if it is specified
            # (and use it further on)
            cflags = sysconfig.get_config_var('CFLAGS')
            m = re.search(r'-isysroot\s+(\S+)', cflags)
            if m is None:
                sysroot = '/'
            else:
                sysroot = m.group(1)

        for dir in dirs:
            shared = os.path.join(dir, shared_f)
            dylib = os.path.join(dir, dylib_f)
            static = os.path.join(dir, static_f)

            if sys.platform == 'darwin' and (
                dir.startswith('/System/') or (
                dir.startswith('/usr/') and not dir.startswith('/usr/local/'))):

                shared = os.path.join(sysroot, dir[1:], shared_f)
                dylib = os.path.join(sysroot, dir[1:], dylib_f)
                static = os.path.join(sysroot, dir[1:], static_f)
            # We can have development libraries names...
            shared_dev = glob.glob(shared+'.*')
            if len(shared_dev) > 0:
                shared_dev = shared_dev[0]
            else:
                shared_dev = None

            dylib_dev  = glob.glob(dylib.replace('dylib', '*.dylib'))
            if len(dylib_dev) > 0:
                dylib_dev = dylib_dev[0]
            else:
                dylib_dev = None

            static_dev = None
            # We're second-guessing the linker here, with not much hard
            # data to go on: GCC seems to prefer the shared library, so I'm
            # assuming that *all* Unix C compilers do.  And of course I'm
            # ignoring even GCC's "-static" option.  So sue me.
            if os.path.exists(dylib):
                return dylib
            elif dylib_dev:
                return dylib_dev
            elif os.path.exists(shared):
                return shared
            elif shared_dev:
                return shared_dev
            elif os.path.exists(static):
                return static

        # Oops, didn't find it in *any* of 'dirs'
        return None

    def find_libraries(self):
        #platform = util.get_platform()
        platform = sys.platform
        lib_dirs = self.compiler.library_dirs

        ldflags = sysconfig.get_config_vars('LDFLAGS')
        for i in ldflags:
            for item in i.split():
                if item.startswith('-L'):
                    lib_dirs.append(item[2:])

        lib_dirs.extend(sysconfig.get_config_vars('LIBDIR'))

        # Add current LD_LIBRARY_PATHs
        env_vars = ['LD_LIBRARY_PATH']
        if platform == 'darwin':
            env_vars =  ['DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']

        for item in env_vars:
            val = os.getenv(item)
            if not val is None:
                dirs = val.split(':')
                lib_dirs.extend(dirs)
        lib_dirs.append(os.path.join(sys.exec_prefix, 'lib'))
        lib_dirs.append(os.path.join(os.getenv('HOME'), 'lib'))
        lib_dirs.extend(['/usr/lib', '/usr/lib'])
        lib_dirs.append('/usr/local/lib')
        # Determine where CCP4 libraries are
        ccp4_dir = os.getenv('CCP4')
        if not ccp4_dir is None:
            lib_dirs.append(os.path.join(ccp4_dir, 'lib'))
        # Check if we have everything
        clipper_core_lib_name = 'clipper-core'
        clipper_core_lib = self._find_library_file(lib_dirs, clipper_core_lib_name)
        if clipper_core_lib is None:
            raise DistutilsSetupError("Could not find CLIPPER libraries. Paths searched %s" % lib_dirs )
        elif platform == 'darwin':
            try:
                # If we have found only dev named library we have to change library name
                suffix = re.findall('%s(.*)\.dylib' % clipper_core_lib_name, clipper_core_lib)[0]
                clipper_core_lib_name += suffix
            except IndexError:
                pass

        clipper_ccp4_lib_name = 'clipper-ccp4'
        clipper_ccp4_lib = self._find_library_file(lib_dirs, clipper_ccp4_lib_name)
        if clipper_ccp4_lib is None:
            raise DistutilsSetupError("Please compile CLIPPER with CCP4 support")
        elif platform == 'darwin':
            try:
                # If we have found only dev named library we have to change library name
                suffix = re.findall('%s(.*)\.dylib' % clipper_ccp4_lib_name, clipper_ccp4_lib)[0]
                clipper_ccp4_lib_name += suffix
            except IndexError:
                pass

        clipper_mmdb_lib_name = 'clipper-mmdb'
        clipper_mmdb_lib = self._find_library_file(lib_dirs, clipper_mmdb_lib_name)
        if clipper_mmdb_lib is None:
            raise DistutilsSetupError("Please compile CLIPPER with MMDB support")
        elif platform == 'darwin':
            try:
                # If we have found only dev named library we have to change library name
                suffix = re.findall('%s(.*)\.dylib' % clipper_mmdb_lib_name, clipper_mmdb_lib)[0]
                clipper_mmdb_lib_name += suffix
            except IndexError:
                pass

        mmdb_lib_name = 'mmdb2'
        mmdb_lib = self._find_library_file(lib_dirs, mmdb_lib_name)
        if mmdb_lib is None:
            raise DistutilsSetupError("Please compile CLIPPER with MMDB support")
        elif platform == 'darwin':
            try:
                # If we have found only dev named library we have to change library name
                suffix = re.findall('%s(.*)\.dylib' % mmdb_lib_name, mmdb_lib)[0]
                mmdb_lib_name += suffix
            except IndexError:
                pass

        ccp4_lib_name = 'ccp4c'
        ccp4_lib = self._find_library_file(lib_dirs, ccp4_lib_name)

        if ccp4_lib is None:
            raise DistutilsSetupError("Could not find CCP4 libraries. Paths searched %s" % lib_dirs)
        elif platform == 'darwin':
            # If we have found only dev named library we have to change library name
            try:
                suffix = re.findall('%s(.*)\.dylib' % ccp4_lib_name, ccp4_lib)[0]
                ccp4_lib_name += suffix
            except IndexError:
                pass

        log.info("Linking against following CLIPPER libraries:\n\t%s\n\t%s\n\t%s\n\t%s"
                       % (clipper_core_lib, clipper_ccp4_lib, clipper_mmdb_lib, mmdb_lib))

        log.info("Linking against following CCP4 library:\n\t%s"
                       % ccp4_lib)

        libs = [ os.path.dirname(l) for l in [clipper_core_lib, 
                                              clipper_ccp4_lib, clipper_mmdb_lib, mmdb_lib, ccp4_lib] ]
        [self.compiler.library_dirs.append(l) for l in libs
         if os.path.exists(l) and l not in self.compiler.library_dirs]
        headers = [ os.path.join(os.path.dirname(l),'include') for l in libs]

        [self.compiler.include_dirs.append(h) for h in headers
         if os.path.exists(h) and h not in self.compiler.include_dirs]

        # Ok first check if user didn't provide multiple libraries in command line
        # Workaround to Python issue #1326113
        # assuming no space in names
        libraries = []
        for item in self.compiler.libraries:
            if ' ' in item:
                libraries.extend(item.split(' '))
            else:
                libraries.append(item)
        self.compiler.libraries = libraries

        self.compiler.libraries.extend([mmdb_lib_name, ccp4_lib_name, clipper_core_lib_name, clipper_mmdb_lib_name, clipper_ccp4_lib_name, 'clipper-contrib'])
        print 'INC', self.compiler.include_dirs
        print 'LIB', self.compiler.libraries

#TODO: Do we need that ?
class build_scripts(_build_scripts):

    wrapper_script = 'scripts/wrapper'
    user_options = _build_scripts.user_options
    user_options.append(
                   ('wrap', None, 'wrap with bash script')
                   )
    boolean_options = _build_scripts.boolean_options
    boolean_options.append('wrap')

    def initialize_options(self):
        _build_scripts.initialize_options(self)
        self.wrap = False
        self.wrap_temp = None

    def finalize_options(self):
        _build_scripts.finalize_options(self)
        wrapper_fn = util.convert_path(self.wrapper_script)
        try:
            wrapper = open(self.wrapper_script, 'r')
        except IOError:
            self.warn("Skipping wrapping: No such file: %s" % wrapper_fn)
            self.wrap = False
        else:
            self.wrapper = wrapper.read()
            if '%%BODY%%' not in self.wrapper:
                self.warn("Skipping wrapping: Malfomed wrapper script (no %%BODY%%)");
                self.wrap = False
        if self.wrap:
            self.set_undefined_options('build',
                                       ('build_temp', 'wrap_temp')
                                       )

    def wrap_scripts(self):
        wrapped_scripts = []
        for script in self.scripts:
            try:
                f = open(script, "r")
            except IOError:
                if not self.dry_run:
                    raise
                f = None
            else:
                data = f.read()
                if not data:
                    self.warn("%s is an empty file (skipping)" % script)
                    continue
                wrapped_data = self.wrapper.replace('%%BODY%%', data)
                outfile = os.path.join(self.wrap_temp, os.path.basename(script))
                wrapped_scripts.append(outfile)
                self.mkpath(self.wrap_temp)
                outf = open(outfile, 'w')
                outf.write(wrapped_data)
                outf.close()

        self.scripts = wrapped_scripts

    def copy_scripts (self):
        if self.wrap:
            self.wrap_scripts()
        _build_scripts.copy_scripts(self)
