#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;
    use sp_runtime::traits::Hash;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    }

    /// Metadata for stored hashes
    #[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub struct HashMetadata<BlockNumber> {
        pub data_type: DataType,
        pub timestamp: BlockNumber,
        pub submitter: [u8; 32],
    }

    /// Types of data that can be hashed
    #[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
    pub enum DataType {
        GIProduct,
        Cluster,
        ROIPrediction,
        MLAScore,
        Anomaly,
        Forecast,
        RiskAssessment,
        ModelPerformance,
    }

    /// Storage map: Hash -> Metadata
    #[pallet::storage]
    #[pallet::getter(fn data_hashes)]
    pub type DataHashes<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        T::Hash,
        HashMetadata<BlockNumberFor<T>>,
        OptionQuery,
    >;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// Hash stored successfully [hash, data_type, block_number]
        HashStored {
            hash: T::Hash,
            data_type: DataType,
            block_number: BlockNumberFor<T>,
        },
        /// Hash verified [hash, exists]
        HashVerified {
            hash: T::Hash,
            exists: bool,
        },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Hash already exists
        HashAlreadyExists,
        /// Hash not found
        HashNotFound,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Store a new data hash
        #[pallet::call_index(0)]
        #[pallet::weight(10_000)]
        pub fn store_hash(
            origin: OriginFor<T>,
            hash: T::Hash,
            data_type: DataType,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            // Ensure hash doesn't already exist
            ensure!(!DataHashes::<T>::contains_key(&hash), Error::<T>::HashAlreadyExists);

            let block_number = <frame_system::Pallet<T>>::block_number();
            
            // Convert AccountId to bytes (simplified)
            let submitter = [0u8; 32]; // In production, properly convert who to bytes

            let metadata = HashMetadata {
                data_type: data_type.clone(),
                timestamp: block_number,
                submitter,
            };

            DataHashes::<T>::insert(&hash, metadata);

            Self::deposit_event(Event::HashStored {
                hash,
                data_type,
                block_number,
            });

            Ok(())
        }

        /// Verify if a hash exists
        #[pallet::call_index(1)]
        #[pallet::weight(10_000)]
        pub fn verify_hash(
            origin: OriginFor<T>,
            hash: T::Hash,
        ) -> DispatchResult {
            let _who = ensure_signed(origin)?;

            let exists = DataHashes::<T>::contains_key(&hash);

            Self::deposit_event(Event::HashVerified {
                hash,
                exists,
            });

            Ok(())
        }
    }
}
