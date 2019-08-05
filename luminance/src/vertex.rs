//! Vertex formats, associated types and functions.
//!
//! A vertex is a type representing a point. It’s common to find vertex positions, normals, colors
//! or even texture coordinates. Even though you’re free to use whichever type you want, you’re
//! limited to a range of types and dimensions. See [`VertexAttribType`] and [`VertexAttribDim`]
//! for further details.

/// A type that can be used as a [`Vertex`] has to implement that trait – it must provide an
/// associated [`VertexDesc`] value via a function call. This associated value gives enough
/// information on the types being used as attributes to reify enough memory data to align and, size
/// and type buffers correctly.
///
/// In theory, you should never have to implement that trait directly. Instead, feel free to use the
/// [luminance-derive] [`Vertex`] proc-macro-derive instead.
///
/// > Note: implementing this trait is `unsafe`.
pub unsafe trait Vertex {
  /// The associated vertex format.
  fn vertex_desc() -> VertexDesc;
}

unsafe impl Vertex for () {
  fn vertex_desc() -> VertexDesc {
    Vec::new()
  }
}

/// A [`VertexDesc`] is a list of [`VertexAttribDesc`]s.
pub type VertexDesc = Vec<VertexBufferDesc>;

/// A vertex attribute descriptor in a vertex buffer.
///
/// Such a description is used to explain what vertex buffers are made of and how they should be
/// aligned / etc.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct VertexBufferDesc {
  pub index: usize,
  pub name: &'static str,
  pub instancing: VertexInstancing,
  pub attrib_desc: VertexAttribDesc
}

impl VertexBufferDesc {
  pub fn new<S>(
    sem: S,
    instancing: VertexInstancing,
    attrib_desc: VertexAttribDesc
  ) -> Self
  where S: Semantics {
    let index = sem.index();
    let name = sem.name();
    VertexBufferDesc { index, name, instancing, attrib_desc }
  }
}

/// Should vertex instancing be used for a vertex attribute?
///
/// Enabling this is done per attribute but if you enable it for a single attribute of a struct, it
/// should be enabled for all others (interleaved vertex instancing is not supported).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VertexInstancing {
  On,
  Off,
}

/// Vertex attribute format.
///
/// Vertex attributes (such as positions, colors, texture UVs, normals, etc.) have all a specific
/// format that must be passed to the GPU. This type gathers information about a single vertex
/// attribute and is completly agnostic of the rest of the attributes used to form a vertex.
///
/// A type is associated with a single value of type [`VertexAttribDesc`] via the [`VertexAttrib`]
/// trait. If such an implementor exists for a type, it means that this type can be used as a vertex
/// attribute.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct VertexAttribDesc {
  /// Type of the attribute. See [`VertexAttribType`] for further details.
  pub ty: VertexAttribType,
  /// Size in bytes that a single element of the attribute takes. That is, if your attribute has
  /// a dimension set to 2, then the unit size should be the size of a single element (not two).
  pub unit_size: usize,
  /// Alignment of the attribute. The best advice is to respect what Rust does, so it’s highly
  /// recommended to use `::std::mem::align_of` to let it does the job for you.
  pub align: usize,
}

/// Possible type of vertex attributes.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum VertexAttribType {
  // signed integral
  Int,
  Int2,
  Int3,
  Int4,
  // unsigned int
  UInt,
  UInt2,
  UInt3,
  UInt4,
  // floating
  Float,
  Float2,
  Float3,
  Float4,
  Float22,
  Float23,
  Float24,
  Float32,
  Float33,
  Float34,
  Float42,
  Float43,
  Float44,
  // boolean
  Bool,
  Bool2,
  Bool3,
  Bool4,
  // array
  //Array(Box<VertexAttribType>, usize)
}

impl VertexAttribType {
  // Check whether the vertex attribute type is signed integral.
  pub fn is_integral(&self) -> bool {
    use VertexAttribType::*;

    match *self {
      Int | Int2 | Int3 | Int4 => true,
      _ => false
    }
  }

  // Check whether the vertex attribute type is unsigned integral.
  pub fn is_unsigned_integral(&self) -> bool {
    use VertexAttribType::*;

    match *self {
      UInt | UInt2 | UInt3 | UInt4 => true,
      _ => false
    }
  }

  // Check whether the vertex attribute type is floating.
  pub fn is_floating(&self) -> bool {
    use VertexAttribType::*;

    match *self {
      Float | Float2 | Float3 | Float4 | Float22 | Float23 | Float24 | Float32 | Float33 | Float42 |
        Float43 | Float44 => true,
      _ => false
    }
  }

  // Check whether the vertex attribute type is boolean..
  pub fn is_boolean(&self) -> bool {
    use VertexAttribType::*;

    match *self {
      Bool | UInt2 | UInt3 | UInt4 => true,
      _ => false
    }
  }


  /// Get the number of unit elements in a vertex attribute.
  ///
  /// That value is always `1` for scalar types, `N` for N-length vectors and `N` for `M×N`
  /// matrices.
  pub fn unit_len(&self) -> usize {
    use VertexAttribType::*;

    match *self {
      Int => 1,
      Int2 => 2,
      Int3 => 3,
      Int4 => 4,
      UInt => 1,
      UInt2 => 2,
      UInt3 => 3,
      UInt4 => 4,
      Float => 1,
      Float2 => 2,
      Float3 => 3,
      Float4 => 4,
      Float22 => 2,
      Float23 => 3,
      Float24 => 4,
      Float32 => 2,
      Float33 => 3,
      Float34 => 4,
      Float42 => 2,
      Float43 => 3,
      Float44 => 4,
      Bool => 1,
      Bool2 => 2,
      Bool3 => 3,
      Bool4 => 4,
      //Array(ref x, _) => x.unit_len(),
    }
  }

  /// Get the number of layers of a vertex attribute.
  ///
  /// That value is always `1` for scalar types and N-length vectors and `M` for `M×N` matrices.
  pub fn layers(&self) -> usize {
    use VertexAttribType::*;

    match *self {
      Int | Int2 | Int3 | Int4 | UInt | UInt2 | UInt3 | UInt4 | Float | Float2 | Float3 | Float4 |
        Bool | Bool2 | Bool3 | Bool4 => 1,
      Float22 => 2,
      Float23 => 2,
      Float24 => 2,
      Float32 => 3,
      Float33 => 3,
      Float34 => 3,
      Float42 => 4,
      Float43 => 4,
      Float44 => 4,
    }
  }
}

/// Class of vertex attributes.
///
/// A vertex attribute type is always associated with a single constant of type [`VertexAttribDesc`],
/// giving GPUs hints about how to treat them.
pub unsafe trait VertexAttrib {
  const VERTEX_ATTRIB_DESC: VertexAttribDesc;
}

/// Vertex attribute semantics.
///
/// Vertex attribute semantics are a mean to make shaders and vertex buffers talk to each other
/// correctly. This is important for several reasons:
///
///   - The memory layout of your vertex buffers might be very different from an ideal case or even
///     the common case. Shaders don’t have any way to know where to pick vertex attributes from, so
///     a mapping is needed.
///   - Sometimes, a shader just need a few information from the vertex attributes. You then want to
///     be able to authorize _“gaps”_ in the semantics so that shaders can be used for several
///     varieties of vertex formats.
///
/// Vertex attribute semantics are any type that can implement this trait. The idea is that
/// semantics must be unique. The vertex position should have an index that is never used anywhere
/// else in the vertex buffer. Because of the second point above, it’s also highly recommended
/// (even though valid not to) to stick to the same index for a given semantics when you have
/// several tessellations – that allows better composition with shaders. Basically, the best advice
/// to follow: define your semantics once, and keep to them.
///
/// > Note: feel free to use the [luminance-derive] crate to automatically derive this trait from
/// > an `enum`.
pub trait Semantics: Sized {
  /// Retrieve the semantics index of this semantics.
  fn index(&self) -> usize;
  /// Get the name of this semantics.
  fn name(&self) -> &'static str;
  /// Get all available semantics.
  fn semantics_set() -> Vec<SemanticsDesc>;
}

impl Semantics for () {
  fn index(&self) -> usize {
    0
  }

  fn name(&self) -> &'static str {
    ""
  }

  fn semantics_set() -> Vec<SemanticsDesc> {
    Vec::new()
  }
}

/// Semantics description.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SemanticsDesc {
  /// Semantics index.
  pub index: usize,
  /// Name of the semantics (used in shaders).
  pub name: String,
}

/// Class of types that have an associated value which type implements [`Semantics`], defining
/// vertex legit attributes.
///
/// Vertex attribute types can be associated with only one semantics.
pub trait HasSemantics {
  type Sem: Semantics;

  const SEMANTICS: Self::Sem;
}

/// A local version of size_of that depends on the state of the std feature.
#[inline(always)]
const fn size_of<T>() -> usize {
  #[cfg(feature = "std")]
  {
    ::std::mem::size_of::<T>()
  }

  #[cfg(not(feature = "std"))]
  {
    ::core::mem::size_of::<T>()
  }
}

/// A local version of align_of that depends on the state of the std feature.
#[inline(always)]
const fn align_of<T>() -> usize {
  #[cfg(feature = "std")]
  {
    ::std::mem::align_of::<T>()
  }

  #[cfg(not(feature = "std"))]
  {
    ::core::mem::align_of::<T>()
  }
}

// Macro to quickly implement VertexAttrib for a given type.
macro_rules! impl_vertex_attribute {
  ($t:ty, $q:ty, $attr_ty:ident) => {
    unsafe impl VertexAttrib for $t {
      const VERTEX_ATTRIB_DESC: VertexAttribDesc = VertexAttribDesc {
        ty: VertexAttribType::$attr_ty,
        unit_size: $crate::vertex::size_of::<$q>(),
        align: $crate::vertex::align_of::<$q>(),
      };
    }
  };
}

// signed integral
impl_vertex_attribute!(i8, i8, Int);
impl_vertex_attribute!(i16, i16, Int);
impl_vertex_attribute!(i32, i32, Int);
impl_vertex_attribute!([i8; 2], i8, Int2);
impl_vertex_attribute!([i8; 3], i8, Int3);
impl_vertex_attribute!([i8; 4], i8, Int4);
impl_vertex_attribute!([i16; 2], i16, Int2);
impl_vertex_attribute!([i16; 3], i16, Int3);
impl_vertex_attribute!([i16; 4], i16, Int4);
impl_vertex_attribute!([i32; 2], i32, Int2);
impl_vertex_attribute!([i32; 3], i32, Int3);
impl_vertex_attribute!([i32; 4], i32, Int4);

// unsigned integral
impl_vertex_attribute!(u8, u8, UInt);
impl_vertex_attribute!(u16, u16, UInt);
impl_vertex_attribute!(u32, u32, UInt);
impl_vertex_attribute!([u8; 2], u8, UInt2);
impl_vertex_attribute!([u8; 3], u8, UInt3);
impl_vertex_attribute!([u8; 4], u8, UInt4);
impl_vertex_attribute!([u16; 2], u16, UInt2);
impl_vertex_attribute!([u16; 3], u16, UInt3);
impl_vertex_attribute!([u16; 4], u16, UInt4);
impl_vertex_attribute!([u32; 2], u32, UInt2);
impl_vertex_attribute!([u32; 3], u32, UInt3);
impl_vertex_attribute!([u32; 4], u32, UInt4);

// floating
impl_vertex_attribute!(f32, f32, Float);
impl_vertex_attribute!([f32; 2], f32, Float2);
impl_vertex_attribute!([f32; 3], f32, Float3);
impl_vertex_attribute!([f32; 4], f32, Float4);
impl_vertex_attribute!([[f32; 2]; 2], f32, Float22);
impl_vertex_attribute!([[f32; 2]; 3], f32, Float23);
impl_vertex_attribute!([[f32; 2]; 4], f32, Float24);
impl_vertex_attribute!([[f32; 3]; 2], f32, Float32);
impl_vertex_attribute!([[f32; 3]; 3], f32, Float33);
impl_vertex_attribute!([[f32; 3]; 4], f32, Float34);
impl_vertex_attribute!([[f32; 4]; 2], f32, Float42);
impl_vertex_attribute!([[f32; 4]; 3], f32, Float43);
impl_vertex_attribute!([[f32; 4]; 4], f32, Float44);
